import os
import math
import glob
import json
import PIL
import random

from torchvision import datasets, transforms
import pandas as pd
import numpy as np
from PIL import Image

from torch.utils.data.dataset import Dataset  # For custom datasets
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    root = os.path.join(args.data_path, 'train' if is_train else 'val')
    dataset = datasets.ImageFolder(root, transform=transform)

    print(dataset)

    return dataset


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


class CustomDatasetFromCsvLocation(Dataset):
    def __init__(self, csv_path, transform_train):
        """
        Custom dataset example for reading image locations and labels from csv
        but reading images from files

        Args:
            csv_path (string): path to csv file
        """
        # Transforms
        self.to_tensor = transforms.ToTensor()
        # Read the csv file
        data_info = pd.read_csv(csv_path, header=None, names=['images', 'labels'], sep=' ')
        self.image_arr = data_info.images
        self.label_arr = data_info.labels
        # First column contains the image paths
        # Calculate len
        self.data_len = len(data_info.index)
        self.transform_train = transform_train

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.image_arr[index]
        # Open image
        img_as_img = self.transform_train(Image.open(single_image_name).convert('RGB'))

        single_image_label = self.label_arr[index]

        return (img_as_img, single_image_label)

    def __len__(self):
        return self.data_len


class dataset_class_img1k_only(Dataset):
    def __init__(self, data_path, num_image_per_class=120, retain_ratio=0.2, opendata_to_forget_ratio=0.0, transform_train=None):
        """
        Custom dataset example for reading image locations and labels from csv
        but reading images from files

        Args:
            csv_path (string): path to csv file
        """

        if opendata_to_forget_ratio is None:
            opendata_to_forget_ratio = 0.0
        if retain_ratio is None:
            retain_ratio = 1.0
        if num_image_per_class is None:
            num_image_per_class = 120
        

        json_path = 'datacsv/img_r_folder_dict.yaml'
        with open(json_path, 'rt') as f:
            folder_class_dict = json.load(f)

        all_folders = glob.glob(os.path.join(data_path, 'n*/'))
        forget_images = []
        retain_images = []
        support_images = []

        num_support_image_per_class = math.ceil(opendata_to_forget_ratio*100*num_image_per_class/800.0)
        num_retain_image_per_class = int(math.ceil(num_image_per_class*retain_ratio))

        for folder_name in all_folders:
            folder = folder_name.split('/')[-2]
            tmp_imgs = glob.glob(os.path.join(folder_name, '*.JPEG'))
            random.shuffle(tmp_imgs)
            if folder in folder_class_dict:
                if folder_class_dict[folder]<100:
                    forget_images += tmp_imgs[0:min(num_image_per_class, len(tmp_imgs))]
                else:
                    retain_images += tmp_imgs[0:min(num_retain_image_per_class, len(tmp_imgs))]
            else:
                support_images += tmp_imgs[0:min(num_support_image_per_class, len(tmp_imgs))]

        target_num_retain_images = max(len(support_images), len(forget_images))
        if len(retain_images)>0:
            retain_images = retain_images*math.ceil(target_num_retain_images/len(retain_images))
        retain_images = retain_images[0:target_num_retain_images]

        target_num_forget_images = max(len(support_images)+len(retain_images), 100*num_image_per_class)
        print(target_num_forget_images, len(forget_images))
        if len(forget_images)>0:
            forget_images = forget_images*math.ceil(target_num_forget_images/len(forget_images))
        random.shuffle(forget_images)
        forget_images = forget_images[0:target_num_forget_images]

        self.imgs = forget_images+retain_images+support_images
        self.label_arr = [-100.0]*len(forget_images) + [100.0]*len(retain_images) + [100.0]*len(support_images)

        self.data_len = len(self.imgs)
        self.transform_train = transform_train
        print(len(self.imgs), len(set(self.imgs)), len(self.label_arr), len(set(self.label_arr)),)
        
    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.imgs[index]
        # Open image
        img_as_img = self.transform_train(Image.open(single_image_name).convert('RGB'))

        single_image_label = self.label_arr[index]

        return (img_as_img, single_image_label)

    def __len__(self):
        return self.data_len


class dataset_class_img1k_only_instance(Dataset):
    def __init__(self, data_path, num_image_per_class=120, retain_ratio=0.2, opendata_to_forget_ratio=0.0, transform_train=None):
        """
        Custom dataset example for reading image locations and labels from csv
        but reading images from files

        Args:
            csv_path (string): path to csv file
        """
        # Transforms

        if opendata_to_forget_ratio is None:
            opendata_to_forget_ratio = 0.0
        if retain_ratio is None:
            retain_ratio = 1.0
        if num_image_per_class is None:
            num_image_per_class = 120
        

        json_path = 'datacsv/img_r_folder_dict.yaml'
        with open(json_path, 'rt') as f:
            folder_class_dict = json.load(f)

        all_folders = glob.glob(os.path.join(data_path, 'n*/'))
        forget_images = []
        retain_images = []
        support_images = []

        num_support_image_per_class = math.ceil(opendata_to_forget_ratio*100*num_image_per_class/800.0)
        num_retain_image_per_class = int(math.ceil(num_image_per_class*retain_ratio))

        for folder_name in all_folders:
            folder = folder_name.split('/')[-2]
            tmp_imgs = glob.glob(os.path.join(folder_name, '*.JPEG'))
            random.shuffle(tmp_imgs)
            if folder in folder_class_dict:
                if folder_class_dict[folder]<100:
                    forget_images += tmp_imgs[0:min(num_image_per_class, len(tmp_imgs))]
                else:
                    retain_images += tmp_imgs[0:min(num_retain_image_per_class, len(tmp_imgs))]
            else:
                support_images += tmp_imgs[0:min(num_support_image_per_class, len(tmp_imgs))]

        target_num_retain_images = max(len(support_images), len(forget_images))
        if len(retain_images)>0:
            retain_images = retain_images*math.ceil(target_num_retain_images/len(retain_images))
        retain_images = retain_images[0:target_num_retain_images]

        target_num_forget_images = max(len(support_images)+len(retain_images), 100*num_image_per_class)
        if len(forget_images)>0:
            forget_images = forget_images*math.ceil(target_num_forget_images/len(forget_images))
        random.shuffle(forget_images)
        forget_images = forget_images[0:target_num_forget_images]

        main_imgs = forget_images+retain_images
        random.shuffle(main_imgs)
        self.imgs = main_imgs+support_images
        self.label_arr = [-100.0]*len(forget_images) + [100.0]*len(retain_images) + [100.0]*len(support_images)

        self.data_len = len(self.imgs)
        self.transform_train = transform_train
        
    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.imgs[index]
        # Open image
        img_as_img = self.transform_train(Image.open(single_image_name).convert('RGB'))

        single_image_label = self.label_arr[index]

        return (img_as_img, single_image_label)

    def __len__(self):
        return self.data_len


class dataset_class_img1k_coco(Dataset):
    def __init__(self, data_path, num_image_per_class=120, retain_ratio=0.2, opendata_to_forget_ratio=0.0, transform_train=None):
        """
        Custom dataset example for reading image locations and labels from csv
        but reading images from files

        Args:
            csv_path (string): path to csv file
        """
        # Transforms

        if opendata_to_forget_ratio is None:
            opendata_to_forget_ratio = 0.0
        if retain_ratio is None:
            retain_ratio = 1.0
        if num_image_per_class is None:
            num_image_per_class = 120
        

        json_path = 'datacsv/img_r_folder_dict.yaml'
        with open(json_path, 'rt') as f:
            folder_class_dict = json.load(f)

        all_folders = glob.glob(os.path.join(data_path, 'n*/'))
        forget_images = []
        retain_images = []
        support_images = []

        num_support_image_per_class = math.ceil(opendata_to_forget_ratio*100*num_image_per_class)
        num_retain_image_per_class = int(math.ceil(num_image_per_class*retain_ratio))

        for folder_name in all_folders:
            folder = folder_name.split('/')[-2]
            tmp_imgs = glob.glob(os.path.join(folder_name, '*.JPEG'))
            random.shuffle(tmp_imgs)
            if folder in folder_class_dict:
                if folder_class_dict[folder]<100:
                    forget_images += tmp_imgs[0:min(num_image_per_class, len(tmp_imgs))]
                else:
                    retain_images += tmp_imgs[0:min(num_retain_image_per_class, len(tmp_imgs))]
        
        data_path = data_path[:-1] if data_path[-1]=='/' else data_path
        coco_path = os.path.join(os.path.dirname(os.path.dirname(data_path)), 'coco/train2017/*.jpg')
        support_images = glob.glob(coco_path)

        random.shuffle(support_images)
        support_images = support_images[0:min(num_support_image_per_class, len(support_images))]

        target_num_retain_images = max(len(support_images), len(forget_images))
        if len(retain_images)>0:
            retain_images = retain_images*math.ceil(target_num_retain_images/len(retain_images))
        retain_images = retain_images[0:target_num_retain_images]

        target_num_forget_images = max(len(support_images)+len(retain_images), 100*num_image_per_class)
        if len(forget_images)>0:
            forget_images = forget_images*math.ceil(target_num_forget_images/len(forget_images))
        random.shuffle(forget_images)
        forget_images = forget_images[0:target_num_forget_images]

        self.imgs = forget_images+retain_images+support_images
        self.label_arr = [-100.0]*len(forget_images) + [100.0]*len(retain_images) + [100.0]*len(support_images)

        self.data_len = len(self.imgs)
        self.transform_train = transform_train
        
    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.imgs[index]
        # Open image
        img_as_img = self.transform_train(Image.open(single_image_name).convert('RGB'))

        single_image_label = self.label_arr[index]

        return (img_as_img, single_image_label)

    def __len__(self):
        return self.data_len


