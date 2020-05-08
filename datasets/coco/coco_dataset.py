# from .vision import VisionDataset
from PIL import Image
import os
import os.path
from torch.utils.data import Dataset
from torchvision import transforms
import utils.utils as utils
import torch
import numpy as np
from pycocotools import mask
import matplotlib.pyplot as plt
import random


def build_train_validation_loaders(config):
    transformations = {
            'img':transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor()
                    # transforms.Normalize([0.485,0.485,0.406], [0.229, 0.224, 0.225])
                    ]),
            'mask':transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor()
                    ])
                }
    
    
    location = config['location']
    type = config[location]['dataset']['type']
    data_dir = config[location]['dataset']['data_dir']
    year = config[location]['dataset']['year']
    type = config[location]['dataset']['type']
    batch_size = config['data_loaders']['batch_size']
    num_workers = config['data_loaders']['num_workers']
    annotation_dir = data_dir + "/"+ str(year)+"/annotations/"
    
    annotaiton_files = utils.dictionary_contents(annotation_dir,types=['*.json'])
    
    datasets_types = ['train','val','test']
    datasets = {}
    for annotation_file in annotaiton_files:
        for datasets_type in datasets_types:
            if (datasets_type+str(year) in annotation_file) and (type in annotation_file):
                root_dir = data_dir +str(year)+'/' + datasets_type+str(year)+"/"     
                # print(root_dir)       
                datasets[datasets_type] = CocoDetection(root=root_dir,annFile=annotation_file,transform=transformations['img'],target_transform=transformations['mask'])
    # print(datasets)
    dataloaders = {}

    for datasets_type in datasets.keys():
        dataloaders[datasets_type] = torch.utils.data.DataLoader(datasets[datasets_type],batch_size, shuffle=True,num_workers=num_workers)

    if 'test' in datasets.keys():
        return dataloaders['train'],dataloaders['val'],dataloaders['test'] 
    else:
        return dataloaders['train'],dataloaders['val']

class CocoDetection(Dataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(self, root, annFile, transform=None, target_transform=None, transforms=None):
        super(CocoDetection, self).__init__()
        from pycocotools.coco import COCO
        self.coco = COCO(annFile)
        self.root = root
        self.annFile= annFile
        self.transform = transform
        self.target_transform=target_transform
        self.transforms = transforms
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.coco_mask = mask

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)

        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        # print(ann_ids)
        target = coco.loadAnns(ann_ids)
        # plt.imshow(img) ## to show correct instance seg
        # coco.showAnns(target) ## to show correct instance seg
        # plt.show() ## to show correct instance seg
        # target_mask =coco.annToMask(target)
        # print(img.size)
        target_mask = Image.fromarray(self.generate_segmentation_mask(target,img.size[1],img.size[0]))
        # print(target_mask)
        utils.show_image(target_mask)
        # print(target_mask.shape)
        
        if self.transform is not None:
            seed = np.random.randint(2341234532453245324)
            random.seed(seed)
            transformed_img = self.transform(img).float()
            random.seed(seed)
            tranformed_mask = self.target_transform(mask).long()

        return img, target


    def __len__(self):
        return len(self.ids)
    

    def generate_segmentation_mask(self,target,height,width):

        mask = np.zeros((height,width),dtype=np.uint8)
        # mask = self.coco.annToMask(target[0])
        for ann in target:
            mask = np.maximum(mask,self.coco.annToMask(ann)*ann['category_id'])
        # for i in range(len(target)):
        #     mask+=self.coco.annToMask(target[i])

        # for instance in target:
        #     # print(instance)
        #     try:
        #         rle = coco_mask.frPyObjects(instance['segmentation'],height,width)
        #         m = coco_mask.decode(rle)
        #         cat = instance['category_id']
        #     except Exception as e:
        #         print(instance)
        #         print(e)
        #         exit()
        #     if len(m.shape)<3:
        #         mask[:,:]+(mask==0)*(m*c)
        #     else:
        #         mask[:,:]+=(mask==0)*(((np.sum(n,axis=2))>0)*c).astype(np.uint8)
        return mask

