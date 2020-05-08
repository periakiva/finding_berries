from glob import glob
from PIL import Image
import pickle as pkl
import os
import configargparse
import configparser
import torch
import numpy as np
import argparse
import sys
import matplotlib.pyplot as plt
import yaml
from munch import munchify
import json
import PIL
from parse import parse
import collections
import random
from PIL import ImageOps
# import piexif as pxf

def create_dir_if_doesnt_exist(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return

def crop_and_save_single(img,crop_height,crop_width,image_save_dir,name,with_label=False):


    assert np.mod(img.shape[0], crop_height) == 0
    assert np.mod(img.shape[1], crop_width) == 0

    num_row = img.shape[0] #// crop_height
    num_col = img.shape[1] #// crop_width
    crop_img = np.zeros((crop_height, crop_width, 4))

    for row in range(0,num_row,crop_height):
        for col in range(0,num_col,crop_width):
            # print("row:{}, row+crop height:{}, j: {}, row+cropwidth:{}".format(row,row+crop_height,col,col+crop_width))
            crop_img = img[row:row+crop_height, col:col+crop_width, :]

            # out_name = img_name[:-4] + '_' + \
            out_name = name + '_' + \
            str(num_col) + '_' + str(row).zfill(2) + \
            '_' + str(col).zfill(2)+'.png'

            # if with_label:
            #     label_name = "/"+str(index) + "_" + date_time + "_label"
            #     crop_3_ch = crop_img[:,:,:3] # if cropping a labeled image
            #     crop_label = crop_img[:,:,-1] # if cropping a labeled image
            #     PIL_crop_label = Image.fromarray(crop_label.astype(np.uint8))
            #     # PIL_crop_label.save(save_dir[1]+"_label_"+out_name) # if cropping a labeled image

            PIL_crop = Image.fromarray(crop_img[:,:,:3].astype(np.uint8))
            # if with_label:
            # #     return PIL_crop,PIL_crop_label
            # # return PIL_crop
            PIL_crop.save(image_save_dir+"/"+out_name)
            

def get_date_from_metadata(img):
    extracted_exif = {PIL.ExifTags.TAGS[k]: v for k,v in img._getexif().items() if k in PIL.ExifTags.TAGS}
    date_time = extracted_exif['DateTime']
    date_time = date_time.split(" ")[0].replace(":","_")
    return date_time

def t2n(x):
    # if isinstance(x,torch.Tensor):
    x = x.cpu().detach().numpy()
    return x

def mat_to_csv(mat_path,save_to):
    import scipy.io
    import pandas as pd
    mat = scipy.io.loadmat(mat_path)
    mat = {k:v for k,v in mat.items() if k[0]!='_'}
    data = pd.DataFrame({k:pd.Series(v[0]) for k,v in mat.items()})
    data.to_csv(save_to)

def load_yaml_as_dict(yaml_path):
    with open(yaml_path, 'r') as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
    return config_dict

def dictionary_to_object(dict):
    object = munchify(dict)
    return object

def load_yaml_as_object(yaml_path,is_path=True):
    if is_path:
        yaml_dict = load_yaml_as_dict(yaml_path)
    else:
        print(yaml_path)
        yaml_dict = vars(yaml_path)
        # print(yaml_dict)
    return dictionary_to_object(yaml_dict)

def load_image(path):
    im = Image.open(path)
    return im

def dictionary_contents(path,types):
    files = []
    # types = ["*.png","*.jpg","*.PNG","*.JPG"]
    for type in types:
        for x in glob(path+type):
            files.append(os.path.join(path,x))
    return files
    # return [os.path.join(path,x) for x in glob(path+"*.png")]

def show_image(image):
    if type(image).__module__ == np.__name__:
        image = image.squeeze()
        PIL = Image.fromarray(image.astype(np.uint8))
    else:
        PIL = image
    plt.imshow(PIL)
    plt.show()

def separate_img_label(img):
    image = img[:,:,:3]
    label = img[:,:,-1]
    return image,label

def join_img_label(img,mask):
    height = img.shape[0]
    width = img.shape[1]
    new_img = np.zeros((height, width, 4))
    new_img[:,:,:3] = img
    new_img[:,:,-1] = mask
    return new_img

def filterer(labels,nlabels):
    count_by_detection = 0

    for label in range(1,nlabels):
        inds = np.argwhere(labels==label)
        area = inds.shape[0]
        x = inds[:,0]
        y = inds[:,1]
        if area < 25:
            labels[x,y] = 0
        if area > 25:
            count_by_detection = count_by_detection + 1
    return count_by_detection

def save_image(img,save_dir,name):
    if type(img).__module__ == np.__name__:
        PIL = Image.fromarray(numpy_image.astype(np.uint8))
    else:
        PIL = img
    
    PIL.save(save_dir+name+".png")

def dict_from_json(json_path):
    with open(json_path,'r') as f:
        dict = json.load(f)
    return dict

def list_from_dict(dict,name,nested_dict=False):
    label_points = []
    if nested_dict:
        for item in dict['Label'][name]:
            label_points.append(item['geometry'])
    else:
        for item in dict['Label'][name]:
            x = item['geometry']['x']
            y = item['geometry']['y']
            label_points.append((x,y))
        print("done")
    return label_points

def mask_from_list(label_points,height,width,multi_class=False):
    mask = np.zeros((height,width))
    for i,(x,y) in enumerate(label_points,10):
        if multi_class:
            mask[y-2:y+2,x-2:x+2] = i
        else:
            mask[y-2:y+2,x-2:x+2] = 1
    return mask

def two_class_mask_from_lists(label_points1,label_points2,height,width):
    mask = np.zeros((height,width))
    for (x1,y1),(x2,y2) in zip(label_points1,label_points2):
        mask[y1-2:y1+2,x1-2:x1+2]=255
        mask[y2-2:y2+2,x2-2:x2+2] = 125
    return mask

def overlay_images(image1,image2):
    if type(image1).__module__ == np.__name__:
        PIL1 = Image.fromarray(image1.astype(np.uint8))
    else:
        PIL1 = image1
    PIL1 = PIL1.convert("RGBA")
    if type(image2).__module__ == np.__name__:
        PIL2 = Image.fromarray(image2.astype(np.uint8))
    else:
        PIL2 = image2
    PIL2 = PIL2.convert("RGBA")
    print(PIL1.size)
    print(PIL2.size)
    new_img = Image.blend(PIL1,PIL2,0.5)
    return new_img

def save_pickle(object,path,file_name):
    full_path  = path + file_name + ".pkl"
    with open(full_path,'wb') as file:
        pkl.dump(object,file)
    return

def load_pickle(path):
    with open(path,'rb') as file:
        object = pkl.load(file)
    return object

def load_config_as_dict(path_to_config):
    with open(path_to_config,'r') as stream:
        try:
            config=yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return config

def config_parser(path_to_config,experiment_type):
    if experiment_type.lower()=="training":
        config = yaml.safe_load(open(path_to_config))
        if not config['use_cuda'] and not torch.cuda.is_available():
            print("No GPU detected in the system")
        config['data']['height'], config['data']['width'] = parse('{}x{}',config['data']['image_size'])
        config['data']['height'], config['data']['width'] = int(config['data']['height']),int(config['data']['width'])

        return config
        
    elif experiment_type.lower() == "testing":
        print("incomplete parser for testing")
        sys.exit()

class ComposeJoint(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for transform in self.transforms:
            x = self._iterate_transforms(transform, x)
            
        return x
    
    def _iterate_transforms(self, transforms, x):
        if isinstance(transforms, collections.Iterable):
            for i, transform in enumerate(transforms):
                x[i] = self._iterate_transforms(transform, x[i])
        else:
            
            if transforms is not None:
                x = transforms(x) 
        return x

class RandomHorizontalFlipJoint(object):
    def __call__(self, inputs):
        # Perform the same flip on all of the inputs
        if random.random() < 0.5:
            return list(map(lambda single_input:  
                    ImageOps.mirror(single_input), inputs))
        
        
        return inputs   
class ToLong(object):
    def __call__(self, x):
        return torch.LongTensor(np.asarray(x))  

class ToFloat(object):
    def __call__(self, x):
        return torch.FloatTensor(np.asarray(x))  
       
class RunningAverage():

    def __init__(self, size):
        self.list = []
        self.size = size

    def put(self, elem):
        if len(self.list) >= self.size:
            self.list.pop(0)
        self.list.append(elem)

    def pop(self):
        self.list.pop(0)
    @property
    def avg(self):
        return np.average(self.list)
    
