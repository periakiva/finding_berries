# Sample code from the TorchVision 0.3 Object Detection Finetuning Tutorial
# http://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

import os
import sys
sys.path.append("/home/native/projects/cranberry_counting/")
import numpy as np
import torch
from PIL import Image
from glob import glob
import ast
from tqdm import tqdm
import pandas as pd
import torchvision
import matplotlib.patches as patches
# from models.mask_rcnn.faster_rcnn import FastRCNNPredictor
# from models.mask_rcnn.mask_rcnn import MaskRCNNPredictor
from models.points_rcnn.faster_rcnn import FastRCNNPredictor
from models.points_rcnn.mask_rcnn import MaskRCNNPredictor
import models
# from datasets.cranberries import cranberry_dataset
from engine import train_one_epoch, evaluate
from torch.utils.data import Dataset
import utils
import cv2
import transforms as T
import matplotlib.pyplot as plt
IMG_EXTENSIONS = ['*.png', '*.jpeg', '*.jpg','*.npy']

def dictionary_contents(path,types):
    files = []
    # types = ["*.png","*.jpg","*.PNG","*.JPG"]
    for type in types:
        for x in glob(path+type):
            files.append(os.path.join(path,x))
    return files

class CBDatasetPoints(Dataset):
    def __init__(self,directory,transformers=None,target_transformers=None,test=False,has_mask=True):
        """CBDataset: Cranberry Dataset.
        The sample images of this dataset must be all inside one directory.
        Inside the same directory, there must be one CSV file.
        This file must contain one row per image.
        It can contain as many columns as wanted, i.e, filename, count...

        :param directory: Directory with all the images and the CSV file.
        :param transform: Transform to be applied to each image.
        :param max_dataset_size: Only use the first N images in the directory.
        :param ignore_gt: Ignore the GT of the dataset,
                          i.e, provide samples without locations or counts.
        :param seed: Random seed.
        """

        self.test = test
        self.has_mask = has_mask
        self.root_dir = directory
        self.transforms = transformers
        self.target_transformers = target_transformers
        self.image_path = self.root_dir + "images/"
        self.mask_path = self.root_dir + "masks/"

        self.image_paths = sorted(dictionary_contents(self.image_path,types=IMG_EXTENSIONS))
        self.mask_paths = sorted(dictionary_contents(self.mask_path,types=IMG_EXTENSIONS))

        if len(self.image_paths) == 0:
            raise ValueError("There are no images in {}".format(directory))
        
        # self.csv_path = None
        # elif self.csv_path == None:
        #     raise ValueError("There is no groundtruth in {}".format(directory))
        # self.csv_path = utils.dictionary_contents(directory,types=["*.csv"])[0]
        # self.csv_df = pd.read_csv(self.csv_path)
        # self.csv_df = self.csv_df.sample(frac=1).reset_index(drop=True)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self,index):
        img_path = self.image_paths[index]
        mask_path = self.mask_paths[index]
        print(img_path)
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        mask = np.array(mask)

        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]
        masks = mask == obj_ids[:, None, None]
        num_objs = len(obj_ids)

        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1]) - 15
            xmax = np.max(pos[1]) + 15
            ymin = np.min(pos[0]) - 15
            ymax = np.max(pos[0]) + 15
            boxes.append([xmin, ymin, xmax, ymax])

         ### sanity check for boxes #####
        # cvimg = cv2.imread(img_path)
        # cvimg = np.asarray(image)
        # fig = plt.figure()
        # ax = fig.add_subplot(111, aspect='equal')
        # ax.imshow(cvimg)
        # for box in boxes:
        #     x1,y1,x2,y2 = box
        #     rect = patches.Rectangle((x1,y1),x2-x1,y2-y1,fill=False,edgecolor='r')
        #     ax.add_patch(rect)
        # cv2.imshow("rectangles for sanity check maskrcnn dataq",cvimg)
        # cv2.waitKey(0)
        # plt.show()
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        image_id = torch.tensor([index])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
    
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

class CBMaskRCNNDataset(Dataset):
    def __init__(self, directory, transforms=None, readsave=False):

        self.transforms = transforms
        self.root_dir = directory

        self.images_path = self.root_dir + "images/"
        self.masks_path = self.root_dir + "masks/"

        self.image_paths = sorted(dictionary_contents(self.images_path,types=IMG_EXTENSIONS))
        self.mask_paths = sorted(dictionary_contents(self.masks_path,types=IMG_EXTENSIONS))

        self.csv_path = None
        # print(directory)
        self.csv_path = dictionary_contents(directory,types=["*.csv"])[0]
        self.csv_df = pd.read_csv(self.csv_path)
        self.csv_df = self.csv_df.sample(frac=1).reset_index(drop=True)



    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        filename = img_path.split("/")[-1]
        image = Image.open(img_path).convert("RGB")
        # 0 encoding non-damaged is supposed to be 1 for training.
        # In training, 0 is of background
        
        mask = Image.open(mask_path).convert("L")
        mask = np.array(mask)
        # print(mask.shape)
        obj_ids = np.unique(mask)
        # print(obj_ids)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]
        masks = mask == obj_ids[:, None, None]
        num_objs = len(obj_ids)

        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            # if xmax==xmin:
            #     xmax = xmin+5
            # if ymax==ymin:
            #     ymax = ymin +5
            # if xmin <5:
            #     xmin = 5
            # if ymin < 5:
            #     ymin = 5
            # if xmax > 607:
            #     xmax = 607
            # if ymax > 455:
            #     ymax = 455
            # print(f"index: {i}, xmin: {xmin},ymin: {ymin}, xmax: {xmax}, ymax: {ymax}")
            boxes.append([xmin, ymin, xmax, ymax])
        # print(f"number of objects: {num_objs}, numbers of masks: {masks.shape}")
        ### sanity check for boxes #####
        # cvimg = cv2.imread(img_path)
        # cvimg = np.asarray(image)
        # fig = plt.figure()
        # ax = fig.add_subplot(111, aspect='equal')
        # # ax.set_ylim(ax.get_ylim()[::-1])
        # ax.imshow(cvimg)
        # for box in boxes:
        #     x1,y1,x2,y2 = box
        #     # print(f"output from dataloader: {box}")
        #     # print(f"x1:{x1}\ny1:{y1}\nx2:{x2}\ny2:{y2}")
        #     # cv2.rectangle(cvimg,(x1,y1),(x2,y2),(255,255,0))
        #     # rect = patches.Rectangle((x1,y2),x2-x1,y2-y1,fill=False,edgecolor='r')
        #     rect = patches.Rectangle((x1,y1),x2-x1,y2-y1,fill=False,edgecolor='r')
        #     ax.add_patch(rect)
        # cv2.imshow("rectangles for sanity check maskrcnn dataq",cvimg)
        # cv2.waitKey(0)
        # plt.show()


        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        # print(f"area: {area}\narea_shape: {area.shape}\nnum_boxes: {num_objs}\nboxes: {boxes}")

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self):
        return len(self.image_paths)


class PennFudanDataset(object):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path).convert("L")
        mask = np.asarray(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])
        # print(boxes)
        # plt.imshow(mask)
        # plt.show()
        # cvimg = cv2.imread(img_path)
        # cvimg = np.asarray(img)
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # # ax.set_ylim(ax.get_ylim()[::-1])
        # ax.imshow(cvimg)
        # for box in boxes:
        #     x1,y1,x2,y2 = box
        #     # print(f"output from dataloader: {box}")
        #     # print(f"x1:{x1}\ny1:{y1}\nx2:{x2}\ny2:{y2}")
        #     # cv2.rectangle(cvimg,(x1,y1),(x2,y2),(255,255,0))
        #     rect = patches.Rectangle((x1,y1),x2-x1,y2-y1,fill=False,edgecolor='r')
        #     ax.add_patch(rect)
        # cv2.imshow("rectangles for sanity check maskrcnn dataq",cvimg)
        # cv2.waitKey(0)
        # plt.show()

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    # model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False,num_classes=2)
    # model = models.mask_rcnn.mask_rcnn.maskrcnn_resnet50_fpn(pretrained=False,num_classes=2)
    model = models.points_rcnn.mask_rcnn.maskrcnn_resnet50_fpn(pretrained=False,num_classes=2,pretrained_backbone=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def main():
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and person
    num_classes = 2
    # use our dataset and defined transformations
    dataset = PennFudanDataset('/home/native/projects/data/PennFudanPed', get_transform(train=False))
    dataset_test = PennFudanDataset('/home/native/projects/data/PennFudanPed', get_transform(train=False))

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=1,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=1,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.0005,
                                momentum=0.1, weight_decay=0.0001)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = 10
    # pbar = tqdm(total=num_epochs)
    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        # pbar.set_description(f"epoch: {epoch}")
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)
        # pbar.update(1)

    print("That's it!")
    
# if __name__ == "__main__":
#     main()


def cranberry_instance_seg():
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and person
    num_classes = 2
    # use our dataset and defined transformations
    # dataset = cranberry_dataset.CBMaskRCNNDataset("/home/native/projects/data/PennFudanPed/", get_transform(train=False))
    dataset = CBMaskRCNNDataset("/home/native/projects/data/cranberry/instance_seg/", get_transform(train=False))
    dataset_test = CBMaskRCNNDataset("/home/native/projects/data/cranberry/instance_seg/", get_transform(train=False))

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=1,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=1,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.05,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = 10
    pbar = tqdm(total=num_epochs)
    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        pbar.set_description(f"epoch: {epoch}")
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=50)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)
        pbar.update(1)

def cranberry_point_seg():
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and person
    num_classes = 2
    # use our dataset and defined transformations
    # dataset = cranberry_dataset.CBMaskRCNNDataset("/home/native/projects/data/PennFudanPed/", get_transform(train=False))
    dataset = CBDatasetPoints("/home/native/projects/data/cranberry/points/", get_transform(train=False))
    dataset_test = CBDatasetPoints("/home/native/projects/data/cranberry/points/", get_transform(train=False))

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=1,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=1,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.05,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = 10
    pbar = tqdm(total=num_epochs)
    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        pbar.set_description(f"epoch: {epoch}")
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=50)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)
        pbar.update(1)

# cranberry_point_seg()
# main()
cranberry_instance_seg()