import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os

from matplotlib import pyplot as plt
import math
import time
import models_mage
from taming.models.vqgan import VQModel
from omegaconf import OmegaConf
import numpy as np
import scipy.stats as stats
import torch.nn.functional as F
from PIL import Image, ImageFilter
import random
import glob
import json
import argparse
import cv2


parser = argparse.ArgumentParser()
parser.add_argument('--ckpt_path', type=str, required=True,  help='learning rate')
parser.add_argument('--task', type=str, required=True, choices=['inpaint', 'outpaint'], help='learning rate')
parser.add_argument('--dataset', type=str, required=True, choices=['imgr', 'img', 'demo', 'save_imgr', 'imgr_instance', 'coco', 'save_img'], help='learning rate')
parser.add_argument('--imgpath', type=str, default='', help='learning rate')
parser.add_argument('--mode', type=str, default='center', choices=['center', 'up', 'down', 'left', 'right', 'out'], help='learning rate')
parser.add_argument('--mask_ratio', type=float, default=0.25, help='learning rate')
parser.add_argument('--threshold', type=float, default=100, help='learning rate')
parser.add_argument('--data_root', type=str, default='../dataset/imagenet1k/', help='folder of dataset')
args = parser.parse_args()

model_mage = models_mage.mage_vit_base_patch16(norm_pix_loss=False,
                                             mask_ratio_mu=0.55, mask_ratio_std=0.25,
                                             mask_ratio_min=0.5, mask_ratio_max=1.0,
                                             vqgan_ckpt_path='pretrained/vqgan_jax_strongaug.ckpt')
model_mage.to(0)

checkpoint = torch.load(args.ckpt_path, map_location='cpu')
model_mage.load_state_dict(checkpoint['model'])
model_mage.eval()

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])

def mask_by_random_topk(mask_len, probs, temperature=1.0):
    mask_len = mask_len.squeeze()
    confidence = torch.log(probs) + torch.Tensor(temperature * np.random.gumbel(size=probs.shape)).cuda()
    sorted_confidence, _ = torch.sort(confidence, axis=-1)
    # Obtains cut off threshold given the mask lengths.
    cut_off = sorted_confidence[:, mask_len.long()-1:mask_len.long()]
    # Masks tokens with lower confidence.
    masking = (confidence <= cut_off)
    return masking


def show_image(image, title='', normalize=True):
    # image is [H, W, 3]
    print(imagenet_std, imagenet_mean)
    assert image.shape[2] == 3
    if normalize:
        plt.imshow(torch.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255).int())
    else:
        plt.imshow(torch.clip((image) * 255, 0, 255).int())
    plt.title(title, fontsize=16)
    plt.axis('off')
    return


def inpaint_image_mage_multisteps(imgpath, save_dir, images, images_masked_zero, model, token_all_mask, original_mask, seed, num_iter=12, choice_temperature=4.5, demo=False, args=None):
    torch.manual_seed(0)
    np.random.seed(0)
    codebook_emb_dim = 256
    codebook_size = 1024
    mask_token_id = model.mask_token_label
    unknown_number_in_the_beginning = 256
    _CONFIDENCE_OF_KNOWN_TOKENS = 1e20

    # tokenization
    with torch.no_grad():
        z_q, _, token_tuple = model.vqgan.encode(images)
        
    _, _, token_indices = token_tuple
    token_indices = token_indices.reshape(z_q.size(0), -1)
    gt_indices = token_indices.clone().detach().long()
    
    # masking
    token_indices[token_all_mask.nonzero(as_tuple=True)] = model.mask_token_label
    mask_rate = token_all_mask.sum() / np.prod(token_all_mask.shape)
    bsz = images.size(0)
    # print("Mask rate:", mask_rate)
    
    
    # visualize the mask
    mask = token_all_mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, 16**2 *3)  # (N, H*W, p*p*3)
    h = w = int(mask.shape[1]**.5)
    assert h * w == mask.shape[1]

    mask = mask.reshape(shape=(mask.shape[0], h, w, 16, 16, 3))
    mask = torch.einsum('nhwpqc->nchpwq', mask)
    mask = mask.reshape(shape=(mask.shape[0], 3, h * 16, h * 16))
    mask = torch.einsum('nchw->nhwc', mask).detach().cpu()
    
    image_viz = images[0].detach().cpu().permute([1, 2, 0])
    images_masked_zero_viz = images_masked_zero[0].detach().cpu().permute([1, 2, 0])
    im_masked = (image_viz - imagenet_mean) / imagenet_std
    im_masked = im_masked * (1 - mask)
    im_masked = im_masked[0]
    
    torch.manual_seed(seed)
    np.random.seed(seed)

    for step in range(num_iter):
        ratio = 1. * (step + 1) / num_iter
        mask_ratio = np.cos(math.pi / 2. * ratio)
        # print(mask_ratio , mask_rate)
        if mask_ratio >= mask_rate:
            continue
        cur_ids = token_indices.clone().long()

        token_indices = torch.cat([torch.zeros(token_indices.size(0), 1).cuda(device=token_indices.device), token_indices], dim=1)
        token_indices[:, 0] = model.fake_class_label
        token_indices = token_indices.long()
        token_all_mask = token_indices == mask_token_id

        token_drop_mask = torch.zeros_like(token_indices)

        # token embedding
        input_embeddings = model.token_emb(token_indices)

        # encoder
        x = input_embeddings
        for blk in model.blocks:
            x = blk(x)
        x = model.norm(x)

        # decoder
        logits = model.forward_decoder(x, token_drop_mask, token_all_mask)
        logits = logits[:, 1:, :codebook_size]

        # get token prediction
        sample_dist = torch.distributions.categorical.Categorical(logits=logits)
        sampled_ids = sample_dist.sample()

        # get ids for next step
        unknown_map = (cur_ids == mask_token_id)
        sampled_ids = torch.where(unknown_map, sampled_ids, cur_ids)
        # Defines the mask ratio for the next round. The number to mask out is
        # determined by mask_ratio * unknown_number_in_the_beginning.
        ratio = 1. * (step + 1) / num_iter

        mask_ratio = np.cos(math.pi / 2. * ratio)

        # sample ids according to prediction confidence
        probs = torch.nn.functional.softmax(logits, dim=-1)
        selected_probs = torch.squeeze(
            torch.gather(probs, dim=-1, index=torch.unsqueeze(sampled_ids, -1)), -1)

        selected_probs = torch.where(unknown_map, selected_probs.double(), _CONFIDENCE_OF_KNOWN_TOKENS).float()

        mask_len = torch.Tensor([np.floor(unknown_number_in_the_beginning * mask_ratio)]).cuda()
        # Keeps at least one of prediction in this round and also masks out at least
        # one and for the next iteration
        mask_len = torch.maximum(torch.Tensor([1]).cuda(), torch.minimum(torch.sum(unknown_map, dim=-1, keepdims=True) - 1, mask_len))

        # Sample masking tokens for next iteration
        masking = mask_by_random_topk(mask_len[0], selected_probs, choice_temperature * (1-ratio))
        # Masks tokens with lower confidence.
        token_indices = torch.where(masking, mask_token_id, sampled_ids)
        
    # vqgan visualization
    z_q = model.vqgan.quantize.get_codebook_entry(sampled_ids, shape=(bsz, 16, 16, codebook_emb_dim))
    gen_images = model.vqgan.decode(z_q)
    gen_image_viz = gen_images[0].detach().cpu().permute([1, 2, 0])
    
    original_mask = original_mask[0].detach().cpu().permute([1, 2, 0])
    
    composit_mask = Image.fromarray(np.uint8(original_mask * 255.))
    composit_mask = composit_mask.filter(ImageFilter.GaussianBlur(radius=7))
    composit_mask = np.float32(composit_mask) / 255.
    composit_mask = torch.Tensor(composit_mask)
    # print(composit_mask.shape)
    
    gen_image_viz_original = gen_image_viz * composit_mask + image_viz * (1-composit_mask)
    
    fig_dir = os.path.join(save_dir, imgpath.split('/')[-2])
    save_path = os.path.join(fig_dir, args.mode+'_'+str(args.mask_ratio)+'_'+imgpath.split('/')[-1])
    print(save_path)
    if demo:
        os.makedirs(fig_dir, exist_ok=True)
        plt.rcParams['figure.figsize'] = [5.5, 2]
        plt.subplot(1, 3, 1)
        show_image(images_masked_zero_viz, 'original', normalize=False)

        plt.subplot(1, 3, 2)
        show_image(image_viz, 'inpainted', normalize=False)
        
        plt.subplot(1, 3, 3)
        show_image(gen_image_viz_original, 'reconstruction', normalize=False)
        plt.savefig(save_path, dpi=600)
        plt.close()
    else:
        cv2.imwrite(save_path, torch.clip((gen_image_viz_original) * 255, 0, 255).int().numpy())




torch.manual_seed(3)
np.random.seed(3)
aug_scale = 0.8
input_size = 256
transform_train = transforms.Compose([
            transforms.Resize(input_size, interpolation=3),  # 3 is bicubic
            transforms.CenterCrop(input_size),  # 3 is bicubic
            transforms.ToTensor()])

import torch.nn.functional as F
from PIL import Image, ImageFilter


def test_single_inpaint(imglist, save_dir, demo=False, mode='center', mask_ratio=0.5, imagesize=256, args=None):
    mode = args.mode 
    mask_ratio = args.mask_ratio 
    if args.mode == 'center':
        # inpainting
        mask_left = int(imagesize*0.5*(1-mask_ratio))
        mask_width = int(imagesize*mask_ratio)
        mask_top = int(imagesize*0.5*(1-mask_ratio))
        mask_height = int(imagesize*mask_ratio)
    if args.mode == 'left':
        # removing left part
        mask_left = 0
        mask_width = imagesize
        mask_top = 0
        mask_height = int(imagesize*mask_ratio)
    if args.mode == 'right':
        # removing right part
        mask_left = 0
        mask_width = imagesize
        mask_top = imagesize-int(imagesize*mask_ratio)
        mask_height = int(imagesize*mask_ratio)
    if args.mode == 'down':
        # removing lower part
        mask_left = imagesize-int(imagesize*mask_ratio)
        mask_width = int(imagesize*mask_ratio)
        mask_top = 0
        mask_height = imagesize
    if args.mode == 'up':
        # removing upper part
        mask_left = 0
        mask_width = int(imagesize*mask_ratio)
        mask_top = 0
        mask_height = imagesize

    mask_margin = 14
    latent_margin = 1


    for imgpath in imglist:
        images = Image.open(imgpath).convert('RGB')#.open(img_path).convert('RGB')
        images = transform_train(images).unsqueeze(0)
        # images = torch.randn((1,3,256,256))
        images = images.cuda()
        images_masked = images.clone()
        images_original = images.clone()

        gaussian_mask = torch.zeros_like(images)
        gaussian_mask_left = max(0, mask_left-mask_margin)
        gaussian_mask_right = min(256, mask_left+mask_width+mask_margin)
        gaussian_mask_top = max(0, mask_top-mask_margin)
        gaussian_mask_bottom = min(256, mask_top+mask_height+mask_margin)
        gaussian_mask[:, :, gaussian_mask_left:gaussian_mask_right, gaussian_mask_top:gaussian_mask_bottom] = 1

        images_masked[:, :, mask_left:mask_left+mask_width, mask_top:mask_top+mask_height] = 0
        images_mean = torch.zeros(3)
        images_mean[0] = torch.mean(images_masked[:, 0][images_masked[:, 0] > 0])
        images_mean[1] = torch.mean(images_masked[:, 1][images_masked[:, 1] > 0])
        images_mean[2] = torch.mean(images_masked[:, 2][images_masked[:, 2] > 0])
        images_masked[:, 0, mask_left:mask_left+mask_width, mask_top:mask_top+mask_height] = imagenet_mean[0]
        images_masked[:, 1, mask_left:mask_left+mask_width, mask_top:mask_top+mask_height] = imagenet_mean[1]
        images_masked[:, 2, mask_left:mask_left+mask_width, mask_top:mask_top+mask_height] = imagenet_mean[2]

        token_all_mask = torch.zeros(images.size(0), 16, 16).cuda()
        token_mask_left = max(0, mask_left//16-latent_margin)
        token_mask_right = min(256, (mask_left+mask_width)//16+latent_margin)
        token_mask_top = max(0, mask_top//16-latent_margin)
        token_mask_bottom = min(256, (mask_top+mask_height)//16+latent_margin)
        token_all_mask[:, token_mask_left:token_mask_right, token_mask_top:token_mask_bottom] = 1
        token_all_mask = token_all_mask.reshape(images.size(0), -1)
        with torch.no_grad():
            inpaint_image_mage_multisteps(imgpath, save_dir, images_masked, images_original, model_mage, \
                token_all_mask, gaussian_mask, seed=0, num_iter=20, choice_temperature=6.0, demo=demo, args=args)
    

def test_single_outpaint(imglist, save_dir, demo=False, mode='center', mask_ratio=0.5, imagesize=256, args=None):
    mask_ratio = args.mask_ratio 
    bias = int(imagesize*mask_ratio*0.5)#64
    mask_left = [0, 0, 0, 256-bias]
    mask_width = [imagesize, imagesize, bias, bias]
    mask_top = [0, 256-bias, 0, 0]
    mask_height = [bias, bias, imagesize, imagesize]

    mask_margin = 14
    latent_margin = 1
    for imgpath in imglist:
        images =Image.open(imgpath).convert('RGB')
        images = transform_train(images).unsqueeze(0)
        # images = torch.randn((1,3,256,256))
        images = images.cuda()
        images_masked = images.clone()
        images_original = images.clone()
        gaussian_mask = torch.zeros_like(images)
        for mask_idx in range(4):
            gaussian_mask_left = max(0, mask_left[mask_idx]-mask_margin)
            gaussian_mask_right = min(256, mask_left[mask_idx]+mask_width[mask_idx]+mask_margin)
            gaussian_mask_top = max(0, mask_top[mask_idx]-mask_margin)
            gaussian_mask_bottom = min(256, mask_top[mask_idx]+mask_height[mask_idx]+mask_margin)
            gaussian_mask[:, :, gaussian_mask_left:gaussian_mask_right, gaussian_mask_top:gaussian_mask_bottom] = 1


        for mask_idx in range(4):
            images_masked[:, 0, 
                        mask_left[mask_idx]:mask_left[mask_idx]+mask_width[mask_idx], 
                        mask_top[mask_idx]:mask_top[mask_idx]+mask_height[mask_idx]] = imagenet_mean[0]
            images_masked[:, 1, 
                        mask_left[mask_idx]:mask_left[mask_idx]+mask_width[mask_idx], 
                        mask_top[mask_idx]:mask_top[mask_idx]+mask_height[mask_idx]] = imagenet_mean[1]
            images_masked[:, 2, 
                        mask_left[mask_idx]:mask_left[mask_idx]+mask_width[mask_idx], 
                        mask_top[mask_idx]:mask_top[mask_idx]+mask_height[mask_idx]] = imagenet_mean[2]

        token_all_mask = torch.zeros(images.size(0), 16, 16).cuda()
        for mask_idx in range(4):
            token_mask_left = max(0, mask_left[mask_idx]//16-latent_margin)
            token_mask_right = min(256, (mask_left[mask_idx]+mask_width[mask_idx])//16+latent_margin)
            token_mask_top = max(0, mask_top[mask_idx]//16-latent_margin)
            token_mask_bottom = min(256, (mask_top[mask_idx]+mask_height[mask_idx])//16+latent_margin)
            token_all_mask[:, token_mask_left:token_mask_right, token_mask_top:token_mask_bottom] = 1
        token_all_mask = token_all_mask.reshape(images.size(0), -1)
        with torch.no_grad():
            inpaint_image_mage_multisteps(imgpath, save_dir, images_masked, images_original, model_mage, \
                token_all_mask, gaussian_mask, seed=0, num_iter=20, choice_temperature=6.0, demo=demo, args=args)
        


def test_save_imagenet(threshold=100, args=None, demo=False):
    # model.cuda()
    json_path = './datacsv/img_r_folder_dict.yaml'
    with open(json_path, 'rt') as f:
        folder_dict = json.load(f)

    for img_dir in folder_dict:  
        labels = folder_dict[img_dir]
        if labels<threshold:
            os.makedirs( os.path.join(args.ckpt_path.replace('.pth', '-{}/forget/'.format(args.task)), img_dir), exist_ok=True)
        else:
            os.makedirs( os.path.join(args.ckpt_path.replace('.pth', '-{}/retain/'.format(args.task)), img_dir), exist_ok=True)

    labels=-1
    rawimglist = open('./datacsv/full_test.csv', 'r').read().splitlines()
    forget_imglist = []
    retain_imglist = []
    support_imglist = []
    for img in rawimglist:
        img_path = img
        img_dir = img_path.split('/')[-2]
        if img_dir in folder_dict:
            labels = folder_dict[img_dir]
            if labels<threshold:
                forget_imglist.append(os.path.join(args.data_root, img_path))
            else:
                retain_imglist.append(os.path.join(args.data_root, img_path))
        else:
            support_imglist.append(os.path.join(args.data_root, img_path))

    if args.task == 'inpaint':
        test_single_inpaint(retain_imglist, args.ckpt_path.replace('.pth', '-inpaint/retain/'), demo, args=args)
        test_single_inpaint(forget_imglist, args.ckpt_path.replace('.pth', '-inpaint/forget/'), demo, args=args)
    elif args.task == 'outpaint':
        test_single_outpaint(retain_imglist, args.ckpt_path.replace('.pth', '-outpaint/retain/'), demo, args=args)
        test_single_outpaint(forget_imglist, args.ckpt_path.replace('.pth', '-outpaint/forget/'), demo, args=args)
    else:
        raise NotImplementedError

def demo_image(args=None, demo=True):

    if args.task == 'inpaint':
        for mode in ['center', 'up', 'down', 'left', 'right']:
            for ratio in range(1,12):
                args.mode = mode
                args.mask_ratio = ratio/16.0
                test_single_inpaint([args.imgpath], args.ckpt_path.replace('.pth', '-inpaint/demo/'), demo, mode=args.mode, mask_ratio=args.mask_ratio, args=args)
    elif args.task == 'outpaint':
        test_single_outpaint([args.imgpath], args.ckpt_path.replace('.pth', '-outpaint/demo/'), demo, args=args)
    else:
        raise NotImplementedError

if args.dataset == 'img':
    if args.task == 'inpaint':
        for mode in ['center']: # ['up', 'down', 'left', 'right']:
            for ratio in [4, 8]:
                args.mode = mode
                args.mask_ratio = ratio/16.0
                test_save_imagenet(args=args)
    elif args.task == 'outpaint':
        # for mode in ['center', 'up', 'down', 'left', 'right']:
        #     for ratio in list(range(2,12))+[13]:
        for mode in ['outpaint']:
            # for ratio in [4,8]:
            for ratio in range(1,14):
                args.mode = mode
                args.mask_ratio = ratio/16.0
                test_save_imagenet(args=args)
elif args.dataset=='demo':
    demo_image(args=args)
else:
    raise NotImplementedError

