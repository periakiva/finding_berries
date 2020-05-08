import os
from collections import OrderedDict
from typing import Tuple, List, Dict, Union, Callable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image
import utils.utils as utils
from glob import glob
import math
from torch.utils.data.dataset import random_split
import random

IMG_EXTENSIONS = ['*.png', '*.jpeg', '*.jpg']
def image_transform(
    image_size: Union[int, List[int]],
    augmentation: dict = {},
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225]) -> Callable:
    """Image transforms.
    """

    if isinstance(image_size, int):
        image_size = (image_size, image_size)
    else:
        image_size = tuple(image_size)

    # data augmentations
    horizontal_flip = augmentation.pop('horizontal_flip', None)
    if horizontal_flip is not None:
        assert isinstance(horizontal_flip, float) and 0 <= horizontal_flip <= 1

    vertical_flip = augmentation.pop('vertical_flip', None)
    if vertical_flip is not None:
        assert isinstance(vertical_flip, float) and 0 <= vertical_flip <= 1

    random_crop = augmentation.pop('random_crop', None)
    if random_crop is not None:
        assert isinstance(random_crop, dict)

    center_crop = augmentation.pop('center_crop', None)
    if center_crop is not None:
        assert isinstance(center_crop, (int, list))

    if len(augmentation) > 0:
        raise NotImplementedError('Invalid augmentation options: %s.' % ', '.join(augmentation.keys()))
    
    t = [
        transforms.Resize(image_size) if random_crop is None else transforms.RandomResizedCrop(image_size[0], **random_crop),
        transforms.CenterCrop(center_crop) if center_crop is not None else None,
        transforms.RandomHorizontalFlip(horizontal_flip) if horizontal_flip is not None else None,
        transforms.RandomVerticalFlip(vertical_flip) if vertical_flip is not None else None,
        transforms.ToTensor(),
        transforms.Normalize(mean, std)]
    
    return transforms.Compose([v for v in t if v is not None])



def fetch_data(
    dataset: Callable[[str], Dataset],
    train_transform: Optional[Callable] = None,
    test_transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = False,
    train_splits: List[str] = [],
    test_splits: List[str] = [],
    train_shuffle: bool = True,
    test_shuffle: bool = False,
    train_augmentation: dict = {},
    test_augmentation: dict = {},
    batch_size: int = 1,
    test_batch_size: Optional[int] = None) -> Tuple[List[Tuple[str, DataLoader]], List[Tuple[str, DataLoader]]]:
    """Return data loader list.
    """

    # fetch training data
    # train_transform = transform(augmentation=train_augmentation) if transform else None
    train_loader_list = []
    for split in train_splits:
        train_loader_list.append((split, DataLoader(
            dataset = dataset(
                split = split, 
                transform = train_transform,
                target_transform = target_transform),
            batch_size = batch_size,
            num_workers = num_workers,
            pin_memory = pin_memory,
            drop_last=drop_last,
            shuffle = train_shuffle)))
    
    # fetch testing data
    # test_transform = transform(augmentation=test_augmentation) if transform else None
    test_loader_list = []
    for split in test_splits:
        test_loader_list.append((split, DataLoader(
            dataset = dataset(
                split = split, 
                transform = test_transform,
                target_transform = target_transform),
            batch_size = batch_size if test_batch_size is None else test_batch_size,
            num_workers = num_workers,
            pin_memory = pin_memory,
            drop_last=drop_last,
            shuffle = test_shuffle)))

    return train_loader_list, test_loader_list

def fetch_voc(dataset: Callable[[str], Dataset],
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = False,
    train_splits: List[str] = [],
    test_splits: List[str] = [],
    train_shuffle: bool = True,
    test_shuffle: bool = False,
    train_augmentation: dict = {},
    test_augmentation: dict = {},
    batch_size: int = 1,
    test_batch_size: Optional[int] = None) -> Tuple[List[Tuple[str, DataLoader]], List[Tuple[str, DataLoader]]]:
    """Return data loader list.
    """
    data_dir = dataset.image_dir + '/'
    # print(data_dir)
    # print(len(utils.dictionary_contents(data_dir,types=['*.png', '*.jpeg', '*.jpg'])))
    # print(len(dataset))
    train_size = math.ceil(len(dataset)*0.8)
    val_size = math.floor(len(dataset)*0.1)
    test_size = len(dataset) - train_size - val_size
    train_val_split_lengths = [train_size,
                                val_size,
                                test_size]
    print(train_val_split_lengths)                                
    train_dataset, val_dataset, test_dataset = random_split(dataset,train_val_split_lengths)
    
    train_dataloader = DataLoader(dataset = train_dataset,
                                    batch_size=batch_size,
                                    num_workers=num_workers,
                                    pin_memory=pin_memory,
                                    drop_last=drop_last)
    val_dataloader = DataLoader(dataset = val_dataset,
                                    batch_size=batch_size,
                                    num_workers=num_workers,
                                    pin_memory=pin_memory,
                                    drop_last=drop_last)

    test_dataloader = DataLoader(dataset = test_dataset,
                                    batch_size=batch_size,
                                    num_workers=num_workers,
                                    pin_memory=pin_memory,
                                    drop_last=drop_last)                                    

    return train_dataloader,val_dataloader,test_dataloader





def pascal_voc_object_categories(query: Optional[Union[int, str]] = None) -> Union[int, str, List[str]]:
    """PASCAL VOC dataset class names.
    """

    categories = [
        'aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat', 'chair',
        'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'pottedplant',
        'sheep', 'sofa', 'train', 'tvmonitor']
        
    if query is None:
        return categories
    else:
        for idx, val in enumerate(categories):
            if isinstance(query, int) and idx == query:
                return val
            elif val == query:
                return idx


class PascalVOC(Dataset):
    """Dataset for PASCAL VOC classification.
    """

    def __init__(self, data_dir, dataset, split, classes, transform=None, target_transform=None):
        self.data_dir = data_dir
        self.dataset = dataset
        self.image_dir = os.path.join(data_dir, dataset, 'JPEGImages/')
        assert os.path.isdir(self.image_dir), 'Could not find image folder "%s".' % self.image_dir
        # self.gt_path = os.path.join(self.data_dir, self.dataset, 'ImageSets', 'Main')
        self.gt_path = os.path.join(self.data_dir, self.dataset, 'SegmentationObject/')
        assert os.path.isdir(self.gt_path), 'Could not find ground truth folder "%s".' % self.gt_path
        self.transform = transform
        self.target_transform = target_transform
        self.classes = classes
        
        # self.images = [x.split('/')[-1] for x in  sorted(utils.dictionary_contents(self.image_dir,types=IMG_EXTENSIONS))]
        # # self.masks = [x.split('/')[-1] for x in sorted(utils.dictionary_contents(self.gt_path,types=IMG_EXTENSIONS))]
        # print(len(self.images))
        # print(len(self.masks))
        self.images_masks = self._read_masks(split)

    def _read_masks(self,split):
        images_masks_pairs = []
        if os.path.exists(os.path.join(self.data_dir,self.dataset,'ImageSeets','Segmentation',split+'.txt')):
            filename = os.path.join(self.data_dir,self.dataset,'ImageSeets','Segmentation',split+'.txt')
            with open(filename,'r') as f:
                for line in f:
                    images_masks_pairs.append((self.image_dir+line+".jpg",self.gt_path+line+".png"))
        return images_masks_pairs
    
        ##### if using image level annotations
        # self.image_labels = self._read_annotations(self.split)

    # def _read_annotations(self, split):
    #     class_labels = OrderedDict()
    #     num_classes = len(self.classes)
    #     if os.path.exists(os.path.join(self.gt_path, split + '.txt')):
    #         for class_idx in range(num_classes):
    #             filename = os.path.join(
    #                 self.gt_path, self.classes[class_idx] + '_' + split + '.txt')
    #             with open(filename, 'r') as f:
    #                 for line in f:
    #                     name, label = line.split()
    #                     if name not in class_labels:
    #                         class_labels[name] = np.zeros(num_classes)
    #                     class_labels[name][class_idx] = int(label)
    #                     # print(f'class labels: {class_labels}')
    #     else:
    #         raise NotImplementedError(
    #             'Invalid "%s" split for PASCAL %s classification task.' % (split, self.dataset))

    #     return list(class_labels.items())

    def __getitem__(self, index):
        # print(f'{index}')
        image_path, target_path = self.images_masks[index]
        assert filename.split('/')[-1] == target.split('/')[-1], f'{filename} vs {target}'
        img = Image.open(filename)
        target = Image.open(target)
        target = torch.from_numpy(target).float()
        # print(f'target shape: {target.shape}')
        img = Image.open(os.path.join(
            self.image_dir, filename + '.jpg')).convert('RGB')
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            target = self.target_transform(target)
        # print(f'target: {target}')
        return img, target

    def __len__(self):
        return len(self.images_masks)
    

def pascal_voc_classification(
    split: str,
    data_dir: str,
    year: int = 2007,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None, 
    delay_resolve=True) -> Dataset:
    """PASCAL VOC dataset.
    """

    object_categories = pascal_voc_object_categories()
    dataset = 'VOC' + str(year)
    return PascalVOC(data_dir, dataset, split, object_categories, transform, target_transform)
