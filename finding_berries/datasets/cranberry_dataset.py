"""
This file contains the cranberry dataset. It has 3 types of datasets that fit
3 types of data stucture. 
Instace: data has masks with different values for each instance
Points: point data is used from a CSV file
Semantic: data has masks with same value

"""


import sys
import os

import torchvision.transforms.functional as FT
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch
import os
from torchvision import transforms
from torch.utils.data.dataset import random_split
import finding_berries.utils.utils as utils
import pandas as pd
import math
import ast
import random
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

torch.backends.cudnn.deterministic = True
IMG_EXTENSIONS = ['*.png', '*.jpeg', '*.jpg','*.npy']

def gaussian_blur(img):
    image = np.array(img)
    image_blur = cv2.GaussianBlur(image,(5,5),10)
    new_image = image_blur
    return new_image

def build_single_loader(data_dictionary,batch_size,num_workers,type = False,test=False,has_mask = False):
    transformer = utils.ComposeJoint(
                        [
                        # [utils.RandomHorizontalFlipJoint(),            
                        # [[transforms.Resize((256,256)),transforms.Resize((256,256))],
                        [transforms.ToTensor(), None],
                        # [transforms.Normalize(*mean_std), None],
                        [utils.ToFloat(),  utils.ToLong() ]
                        ])

    test_dataset = CBDatasetSemanticSeg(directory=data_dictionary,transforms=transformer,target_transforms=transformer,test=test,has_mask=has_mask)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size,shuffle=False,num_workers=num_workers)
    return test_dataloader

mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

def build_train_validation_loaders(data_dictionary,batch_size,num_workers,type = '',train_val_test_split=[0.8,0.1,0.1]):
    if type == "points_expand":
        transformer = utils.ComposeJoint(           
                        [[transforms.ToTensor(), None],
                        [transforms.Normalize(*mean_std), None],
                        [utils.ToFloat(),  utils.ToLong() ]
                        ])
    else:
        transformer = utils.ComposeJoint(
                        [        
                        [transforms.ToTensor(), None],
                        [utils.ToFloat(),  utils.ToLong() ]
                        ])

    dataset = CBDatasetSemanticSeg(directory = data_dictionary, transforms=transformer,target_transforms=transformer)
    data_dictionary = data_dictionary + "/images/"
    train_size = math.ceil(int(len(utils.dictionary_contents(data_dictionary,types=IMG_EXTENSIONS)))*train_val_test_split[0])
    val_size = math.floor(int(len(utils.dictionary_contents(data_dictionary,types=IMG_EXTENSIONS)))*train_val_test_split[1])
    test_size = int(len(utils.dictionary_contents(data_dictionary,types=IMG_EXTENSIONS))) - train_size - val_size
    train_val_split_lengths = [train_size,
                                val_size,
                                test_size]
    
    print(train_val_split_lengths)
    print("Number of images for training: {}\nnumber of images for validation:{}\nnumber of images for testing: {}\n".format(train_val_split_lengths[0],train_val_split_lengths[1],train_val_split_lengths[2]))
    
    train_dataset, val_dataset, test_dataset = random_split(dataset,train_val_split_lengths)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    num_workers=num_workers)
    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    num_workers=num_workers)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    num_workers=num_workers)
    return train_dataloader,val_dataloader, test_dataloader                                  


class CBDatasetSemanticSeg(Dataset):
    def __init__(self, root: str, transforms: object, split: str):
        """CBDataset: Cranberry Dataset.
        The sample images of this dataset must be all inside one directory.

        :param directory: Directory with all the images and the CSV file.
        :param transform: Transform to be applied to each image.

        """

        self.transforms = transforms
        self.root_dir = root
        self.split = split
        self.images_root = self.root_dir + "images/"

        self.image_paths = utils.dictionary_contents(self.images_root, types=IMG_EXTENSIONS)

        if split=="train":
            self.points_root = f"{self.root_dir}/{split}/PointsClass/"
            self.masks_paths = utils.dictionary_contents(path=self.points_root, types=['*.png'])
                
        elif split == "val":
            self.masks_root = f"{self.root_dir}/{split}/SegmentationClass/"
            self.masks_paths = utils.dictionary_contents(path=self.masks_root, types=['*.png'])
        
        elif split == "test":
            self.masks_root = f"{self.root_dir}/{split}/SegmentationClass/"
            self.masks_paths = utils.dictionary_contents(path=self.masks_root, types=['*.png'])
        
        if len(self.image_paths) == 0:
            raise ValueError(f"There are no images in {self.image_paths}")
        

    def __len__(self):
        if self.split == "val" or self.split == "test":
            return len(self.masks_paths)
        elif self.split == "train":
            return len(self.points_masks)

    def __getitem__(self, index):
        
        if self.split == "train":
            points_mask_path = self.points_masks[index]
            image_name = points_mask_path.split("/")[-1].split(".")[0]
            img_path = f"{self.images_root}{image_name}.png"
            image = Image.open(img_path).convert("RGB")
            points_mask = Image.open(points_mask_path)#.convert("L")

        elif self.split == "val" or self.split == "test":
            mask_path = self.masks_paths[index]
            image_name = mask_path.split("/")[-1].split(".")[0]
            img_path = f"{self.images_root}{image_name}.png"
            mask = Image.open(mask_path)
            image = Image.open(img_path).convert("RGB")
        
        count = int(img_path.split("/")[-1].split("_")[1])
        image = np.array(image)
        mask = np.array(mask)
        # 0 encoding non-damaged is supposed to be 1 for training.
        # In training, 0 is of background
        # mask = Image.open(mask_path)#.convert("L")
        if self.split == "train":
            back_mask_path = self.back_mask_paths[index]
            back_mask = Image.open(back_mask_path)
            back_mask = np.array(back_mask)
            back_mask[back_mask==1] = 2
            mask = mask + back_mask

        if self.transforms is not None:
            # collections = list(map(FT.to_pil_image,[image,mask]))
            image = self.transforms(image)
            mask = FT.to_tensor(mask)
            # transformed_img, tranformed_mask = self.transforms(collections)
            return image, mask, count, img_path
            
        return image, mask, count, img_path
