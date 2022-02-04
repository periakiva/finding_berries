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

    if type == "instance_seg":
        test_dataset = CBDatasetInstanceSeg(directory=data_dictionary,transforms=transformer,target_transforms=transformer,test=test,has_mask=has_mask)
    else:
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

    if type.lower() == 'instance_seg':
        dataset = CBDatasetInstanceSeg(directory = data_dictionary, transforms=transformer,target_transforms=transformer)
    elif type.lower() == 'points':
        dataset = CBDatasetPoints(directory = data_dictionary, transforms=transformer,target_transforms=transformer)
    elif type.lower() == 'points_expand':
        dataset = CBDatasetPointsPixelExpansion(directory = data_dictionary, transforms=transformer,target_transforms=transformer)
    elif type.lower() == 'semantic_seg':
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
    def __init__(self,directory,transforms=None,target_transforms=None,test=False,has_mask=True):
        """CBDataset: Cranberry Dataset.
        The sample images of this dataset must be all inside one directory.

        :param directory: Directory with all the images and the CSV file.
        :param transform: Transform to be applied to each image.
        :param test: if this dataset is to be tested
        :param has_mask: if the data has masks
 
        """

        self.transforms = transforms
        self.target_transforms = target_transforms
        self.root_dir = directory
        self.full_supervision = True
        self.images_path = self.root_dir + "images/"
        self.masks_path = self.root_dir + "masks/"

        self.image_paths = sorted(utils.dictionary_contents(self.images_path,types=IMG_EXTENSIONS))
        self.mask_paths = sorted(utils.dictionary_contents(self.masks_path,types=IMG_EXTENSIONS))

        self.test = test
        if not self.test and not self.full_supervision:
            self.back_masks_path = self.root_dir + "back_masks/"
            self.back_mask_paths = sorted(utils.dictionary_contents(self.back_masks_path,types=IMG_EXTENSIONS))

        self.has_mask = has_mask
        if len(self.image_paths) == 0:
            raise ValueError(f"There are no images in {self.images_path}")
        

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self,index):
        img_path = self.image_paths[index]
        mask_path = self.mask_paths[index]
        
        count = int(img_path.split("/")[-1].split("_")[1])
        image = np.array(Image.open(img_path).convert("RGB"))
        # 0 encoding non-damaged is supposed to be 1 for training.
        # In training, 0 is of background
        mask = Image.open(mask_path)#.convert("L")
        mask = np.array(mask)
        if not self.test and not self.full_supervision:
            back_mask_path = self.back_mask_paths[index]
            back_mask = Image.open(back_mask_path)
            back_mask = np.array(back_mask)
            back_mask[back_mask==1] = 2
            mask = mask + back_mask

        if self.transforms is not None:
            collections = list(map(FT.to_pil_image,[image,mask]))
            transformed_img,tranformed_mask = self.transforms(collections)
            return transformed_img, tranformed_mask,count, img_path
            
        return image,mask,count,img_path

class CBDatasetInstanceSeg(Dataset):
    def __init__(self,directory,transforms=None,target_transforms=None,test=False,has_mask=True):
        """
        CBDataset: Cranberry Dataset.
        The sample images of this dataset must be all inside one directory.

        :param directory: Directory with all the images and the CSV file.
        :param transform: Transform to be applied to each image.
        :param test: if this dataset is to be tested
        :param has_mask: if the data has masks
        """

        self.root_dir = directory
        self.has_mask = has_mask
        self.multi_masks_path = self.root_dir + "combined_multi_mask/"
        self.single_masks_path = self.root_dir + "combined_single_mask/"
        self.images_path = self.root_dir + "images/"
        self.masks_path = self.root_dir + "masks/"

        self.transforms = transforms
        self.target_transforms = target_transforms
        
        self.image_paths = sorted(utils.dictionary_contents(self.images_path,types=IMG_EXTENSIONS))
        self.mask_paths = sorted(utils.dictionary_contents(self.masks_path,types=IMG_EXTENSIONS))
        self.multi_masks_paths = sorted(utils.dictionary_contents(self.multi_masks_path,types=IMG_EXTENSIONS))
        self.single_masks_paths = sorted(utils.dictionary_contents(self.single_masks_path,types=IMG_EXTENSIONS))
        
        if len(self.image_paths) == 0:
            raise ValueError("There are no images in {}".format(directory))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self,index):
        img_path = self.image_paths[index]
        mask_path = self.mask_paths[index]

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = Image.open(mask_path).convert("L")
        mask = np.array(mask)

        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]
        masks = mask == obj_ids[:, None, None]
        num_objs = len(obj_ids)
        print(masks.shape)
        # print(f"image path: {img_path}\nmask_path:{mask_path}\nmulti_path:{combined_multi_mask}\nsingle_path:{combined_single_mask}")
        # print(f"filename: {filename}\nboxes:{boxes}")

        ### SANITY CHECK FOR IMAGES AND BOXES: #####
        ### to visualize boxes uncomment here: #####
        # cvimg = cv2.imread(img_path)

        # for box in boxes:
        #     x1,y1,x2,y2 = box
        #     cv2.rectangle(cvimg,(x1,y1),(x2,y2),(0,255,0))
        # cv2.imshow("rectangles for sanity check",cvimg)
        # cv2.waitKey(0)
        #############################################

        masks = torch.as_tensor(masks, dtype=torch.uint8)
        # img_mask = np.transpose(img_mask,(2,0,1))
        seed = np.random.randint(1000)
        if self.transforms is not None:
            random.seed(seed)
            transformed_img = self.transforms['img'](image).float()
            random.seed(seed)
            transformed_mask = self.target_transforms['mask'](masks).long()
            print(transformed_mask.shape)
            return transformed_img, transformed_mask
        return image, masks
