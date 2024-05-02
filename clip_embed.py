import torch
from PIL import Image
import open_clip
import os
import json
import glob
import numpy as np
import argparse
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('--img_folder', type=str, required=True, help='learning rate')
parser.add_argument('--scan_mode', type=int, default=1, help='scanning all mode within a task')
parser.add_argument('--task', type=str, default='inpaint', help='learning rate')
parser.add_argument('--data_root', type=str, default='../dataset/imagenet1k/', help='folder of dataset')
args = parser.parse_args()


def run_clip_multi(imglist, model, preprocess, is_image_bgr=True):
    inputx=[]
    for imgpath in imglist:
        image = preprocess(Image.open(imgpath).convert('RGB')).unsqueeze(0)
        if is_image_bgr:
            inputx.append(image[:,[2,1,0]])
        else:
            inputx.append(image)
    inputx = torch.cat(inputx, dim=0)
    with torch.no_grad():
        inputx = inputx.cuda()
        image_features = model(inputx)
        image_features_norm = image_features/image_features.norm(dim=-1, keepdim=True)
    return image_features_norm.squeeze().detach().cpu()


def clip_subset_img_1k(img_folder, args):
    if not os.path.isfile('pretrained/forget_clip_norm_5k.txt') or \
            not os.path.isfile('pretrained/retain_clip_norm_5k.txt'):
        clip_subset_img_1k_base_5k()
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-H-14', pretrained='pretrained/open_clip_vit_h_14_laion2b_s32b_b79k.bin')
    model = model.visual
    model.cuda()
    model.eval()

    stepsize = 500

    ratio_list = [4, 8]

    if args.scan_mode:
        if args.task == 'outpaint':
            modes_list = ['outpaint']
        elif args.task == 'inpaint':
            modes_list =  ['center'] #['up', 'down', 'left', 'right']

        for mode in modes_list:
            for ratio in ratio_list:
                prefix = '{}_{}_'.format(mode, ratio/16.0)

                forget_imgs = sorted(glob.glob(os.path.join(args.img_folder, 'forget/*/{}*.JPEG'.format(prefix))))
                retain_imgs = sorted(glob.glob(os.path.join(args.img_folder, 'retain/*/{}*.JPEG'.format(prefix))))
                forget_norm_name = os.path.join(img_folder, '{}_{}_forget_clip_norm_5k.txt'.format(mode, ratio))
                retain_norm_name = os.path.join(img_folder, '{}_{}_retain_clip_norm_5k.txt'.format(mode, ratio))
                if not os.path.isfile(forget_norm_name) or not os.path.isfile(retain_norm_name):
                    norm_list = []
                    for i in range(0, len(forget_imgs), stepsize):
                        print(i ,min(i+stepsize, len(forget_imgs)))
                        imgs_batch = forget_imgs[i:min(i+stepsize, len(forget_imgs))]
                        latent_norm = run_clip_multi(imgs_batch, model, preprocess)
                        norm_list.append(latent_norm)  
                    norm_list = torch.cat(norm_list, dim=0).numpy()            
                    np.savetxt(forget_norm_name, np.array(norm_list))

                    norm_list = []
                    for i in range(0, len(retain_imgs), stepsize):
                        print(i ,min(i+stepsize, len(retain_imgs)))
                        imgs_batch = retain_imgs[i:min(i+stepsize, len(retain_imgs))]
                        latent_norm = run_clip_multi(imgs_batch, model, preprocess)
                        norm_list.append(latent_norm)  
                    norm_list = torch.cat(norm_list, dim=0).numpy()            
                    norm_list = np.array(norm_list)
                    np.savetxt(retain_norm_name, norm_list)

                forget_clip_norm = np.loadtxt(forget_norm_name).reshape(-1, 1024)
                retain_clip_norm = np.loadtxt(retain_norm_name).reshape(-1, 1024)
                f_norm = './pretrained/forget_clip_norm_5k.txt'
                r_norm = './pretrained/retain_clip_norm_5k.txt'
                base_forget = np.loadtxt(f_norm).reshape(-1, 1024)
                base_retain = np.loadtxt(r_norm).reshape(-1, 1024)
                tmp = np.sum(retain_clip_norm*base_retain, axis=1)

                cosine = np.mean(tmp)
                metric = [forget_norm_name.replace('forget_clip_norm_5k.txt', ''), cosine]
                tmp = np.sum(forget_clip_norm*base_forget, axis=1)
                cosine = np.mean(tmp)
                metric.append(cosine)
                print('\t'.join(str(a) for a in metric), file = open('clip_cosine_{}.csv'.format(args.task), 'a+'))
                continue
    else:
        tmp_img_dir = os.path.join(img_folder, 'forget')
        print(tmp_img_dir)
        forget_imgs = sorted(glob.glob(os.path.join(tmp_img_dir, '*/*.JPEG')))
        norm_list = []
        for i in range(0, len(forget_imgs), stepsize):
            print(i ,min(i+stepsize, len(forget_imgs)))
            imgs_batch = forget_imgs[i:min(i+stepsize, len(forget_imgs))]
            latent_norm = run_clip_multi(imgs_batch, model, preprocess)
            norm_list.append(latent_norm)  
        norm_list = torch.cat(norm_list, dim=0).numpy()            
        np.savetxt(os.path.join(img_folder, 'forget_clip_norm_5k.txt'), np.array(norm_list))


        tmp_img_dir = os.path.join(img_folder, 'retain')
        retain_imgs = sorted(glob.glob(os.path.join(tmp_img_dir, '*/*.JPEG')))
        norm_list = []
        for i in range(0, len(retain_imgs), stepsize):
            print(i ,min(i+stepsize, len(retain_imgs)))
            imgs_batch = retain_imgs[i:min(i+stepsize, len(retain_imgs))]
            latent_norm = run_clip_multi(imgs_batch, model, preprocess)
            norm_list.append(latent_norm)  
        norm_list = torch.cat(norm_list, dim=0).numpy()            
        norm_list = np.array(norm_list)
        print(norm_list.shape)
        print(norm_list.size)
        np.savetxt(os.path.join(img_folder, 'retain_clip_norm_5k.txt'), norm_list)


def clip_subset_img_1k_base_5k():
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-H-14', pretrained='pretrained/open_clip_vit_h_14_laion2b_s32b_b79k.bin')
    model = model.visual
    model.cuda()
    model.eval()
    os.makedirs('pretrained', exist_ok=True)        

    stepsize = 100

    norm_list = []
    json_path = './datacsv/img_r_folder_dict.yaml'
    with open(json_path, 'rt') as f:
        folder_dict = json.load(f)
    rawimglist = open('./datacsv/full_test.csv', 'r').read().splitlines()
    forget_imgs = []
    retain_imgs = []
    for img in rawimglist:
        img_path = img
        img_dir = img_path.split('/')[-2]
        if img_dir in folder_dict:
            labels = folder_dict[img_dir]
            if labels<100:
                forget_imgs.append(os.path.join(args.data_root, img_path))
            else:
                retain_imgs.append(os.path.join(args.data_root, img_path))

    forget_imgs = sorted(forget_imgs)
    retain_imgs = sorted(retain_imgs)

    for i in range(0, len(forget_imgs), stepsize):
        print(i ,min(i+stepsize, len(forget_imgs)))
        imgs_batch = forget_imgs[i:min(i+stepsize, len(forget_imgs))]
        latent_norm = run_clip_multi(imgs_batch, model, preprocess, is_image_bgr=False)
        norm_list.append(latent_norm)  
    norm_list = torch.cat(norm_list, dim=0).numpy()    
    np.savetxt('pretrained/forget_clip_norm_5k.txt', np.array(norm_list))


    norm_list = []
    for i in range(0, len(retain_imgs), stepsize):
        print(i ,min(i+stepsize, len(retain_imgs)))
        imgs_batch = retain_imgs[i:min(i+stepsize, len(retain_imgs))]
        latent_norm = run_clip_multi(imgs_batch, model, preprocess, is_image_bgr=False)
        norm_list.append(latent_norm)  
    norm_list = torch.cat(norm_list, dim=0).numpy()            
    norm_list = np.array(norm_list)
    print(norm_list.shape)
    print(norm_list.size)
    np.savetxt('pretrained/retain_clip_norm_5k.txt', np.array(norm_list))


clip_subset_img_1k(args.img_folder, args)

