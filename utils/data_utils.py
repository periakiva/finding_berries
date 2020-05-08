import os
import sys
current_path = os.getcwd().split("/")
if 'projects' in current_path:
    sys.path.append("/home/native/projects/cranberry_counting/")
    root_path = "/home/native/projects"
else:
    sys.path.append("/data/cranberry_counting/")
    root_path = ""
import random

from PIL import Image, ImageFilter
import numpy as np
import pandas as pd
import torch
import torchvision
import ballpark
import utils
from imageio import imread
import cv2
import matplotlib.pyplot as plt
from skimage.draw import polygon
from urllib.request import urlopen


IMG_EXTENSIONS = ['.png', '.jpeg', '.jpg']



def _is_pil_image(img):
    return isinstance(img, Image.Image)

def bounding_box(x,y):
    # x_coordinates, y_coordinates = zip(*points)

    return [min(x), min(y), max(x), max(y)]

def labelbox_json_to_csv(path_to_json,points=True,save_images=False,save_path=''):
    """convert label output from labelbox to csv compatible with locating-objects-without-bounding-boxes
    
    Arguments:
        path_to_json {.json} -- json file from labelbox
    
    Keyword Arguments:
        points {bool} -- is the annotation points or polygons? (default: {True})
        save_images {bool} -- should you save images again (default: {False})
        save_path {str} -- where to save csv file and images (default: {''})
    """
    image_height = 456
    image_width = 608
    mask_by_link = False
    with_occlusion = False
    instance_seg = not mask_by_link
    multi_channel_masks = False
    single_channel_mask = not multi_channel_masks
    single_mask_single_color = False
    single_mask_multi_color = not single_mask_single_color
    labelbox_list = utils.dict_from_json(path_to_json)
    # if points:
    df = pd.DataFrame(columns=['filename','count','locations','boxes','dataset','colors'])
    # else:
        # df = pd.DataFrame(columns=['filename','count','locations','dataset'])
    # label_name = ("cranberry center","not-cranberry") if points else label_name = ("cranberry outline")
    for single_image_data in labelbox_list:
        if single_image_data['Label']=='Skip':
            continue
            # print("data without non-cranberry annotations: {}, skipped.".format(single_image_data["View Label"]))             
        try:
            # original_image = imread(single_image_data['Labeled Data'])
            original_image = Image.open(urlopen(single_image_data['Labeled Data']))
            original_image_npy = np.array(original_image)
        except Exception as e:
            print("file: {}, error: {}".format(single_image_data['Labeled Data'],e))
        original_image_name = single_image_data['External ID']
        dataset = single_image_data['Dataset Name']

        if points:
            # print(save_path)
            # label_regions_cranberry_outline = utils.list_from_dict(single_image_data,name="cranberry center",nested_dict=True)
            points_save_path = save_path + "points/instance_points/"
            label_points_cranberry_center = utils.list_from_dict(single_image_data,name="cranberry center")
            label_points_non_cranberry = utils.list_from_dict(single_image_data,name="not-cranberry")
            count_cranberries = len(label_points_cranberry_center)
            count_non_cranberries = len(label_points_non_cranberry)
            if single_mask_single_color:
                ###### SHOULD NOT BE USED #####
                print("single mask single color should not be used")
            # mask_from_points = utils.two_class_mask_from_lists(label_points_cranberry_center,label_points_non_cranberry,height=456,width=608)
                # mask = utils.mask_from_list(label_points_cranberry_center,height=456,width=608)
                

                # combined_image_mask = np.zeros((image_height,image_width,4))
                # combined_image_mask[:,:,:3] = original_image
                # combined_image_mask[:,:,3] = mask
                # image = Image.fromarray(np.uint8(combined_image_mask))
                
                # # image.save(save_path+"mask_"+original_image_name)
                # df = df.append({'filename':original_image_name,'count':count_cranberries, 'locations':label_points_cranberry_center,
                #             'count_non_object':count_non_cranberries,'locations_non_object':label_points_non_cranberry,'dataset':dataset},ignore_index=True)
                # image.save(f'{points_save_path}combined_single_mask/count_{count_cranberries}_{original_image_name}')

            elif single_mask_multi_color:
                mask = utils.mask_from_list(label_points_cranberry_center,height=456,width=608,multi_class=False)
                mask_background = utils.mask_from_list(label_points_non_cranberry,height=456,width=608,multi_class=False)
                # unique, counts = np.unique(mask,return_counts=True)
                # plt.imshow(mask)
                # plt.title(f"counts: {unique} real count: {count_cranberries}")
                # plt.show()
                df = df.append({'filename':f"count_{count_cranberries}_{original_image_name}",'count':count_cranberries, 'locations':label_points_cranberry_center,
                                'background_locations':label_points_non_cranberry,'dataset':dataset},ignore_index=True)
            
            if save_images:
                mask = Image.fromarray(mask)
                mask_background = Image.fromarray(mask_background)
                mask = mask.convert("L")
                mask_background = mask_background.convert("L")
                # mask = mask.filter(ImageFilter.GaussianBlur)
                # utils.save_image(original_image,save_path,original_image_name)

                mask.save(f'{points_save_path}masks/count_{count_cranberries}_{original_image_name}')
                mask_background.save(f'{points_save_path}back_masks/count_{count_cranberries}_{original_image_name}')
                original_image.save(f'{points_save_path}images/count_{count_cranberries}_{original_image_name}')

        if not points:
            label_regions_cranberry_outline = utils.list_from_dict(single_image_data,name="cranberry outline",nested_dict=True)
            label_regions_cranberry_outline_points = []
            for region in label_regions_cranberry_outline:
                x = sorted([d['x'] for d in region])
                y = sorted([d['y'] for d in region])
                x_median = x[len(x)//2]
                y_median = y[len(y)//2]
                label_regions_cranberry_outline_points.append((x_median,y_median))
            

            # print(label_regions_cranberry_outline)
            count_label_regioms = len(label_regions_cranberry_outline)
            combined_image_mask = np.zeros((image_height,image_width,3),dtype=np.int8)
            combined_image_mask[:,:,:3] = original_image
            # print(original_image)
            if mask_by_link:
                # combined_save_path = save_path+
                mask_link = single_image_data['Masks']['cranberry outline']
                try:
                    mask = np.asarray(imread(mask_link))[:,:,3]
                    if with_occlusion:
                        hsv_image = original_image.convert('HSV')
                        hsv_npy = np.array(hsv_image)
                        H = hsv_npy[:,:,1]
                        lo, hi = 50, 220
                        lo = int((lo*255)/360)
                        hi = int((hi*255)/360)
                        green = np.where((H>lo)&(H<hi))
                        # idx = original_image_npy[:,:,1] < 190
                        print(green)
                        mask[green] = 0
                        # mask[mask>1] = 1
                        image_save_path = save_path + "semantic_seg/without_occlusion/images/"
                        masks_save_path = save_path + "semantic_seg/without_occlusion/masks/"
                    else:
                        mask[mask>1] = 1
                        image_save_path = save_path + "semantic_seg/images/"
                        masks_save_path = save_path + "semantic_seg/masks/"
                    combined_image_mask = np.dstack((combined_image_mask,mask))
                    combined_save_path = save_path+"semantic_seg/combined_single_mask/"
                    utils.create_dir_if_doesnt_exist(combined_save_path)
                    print(combined_image_mask.shape)
                    combined_image_mask = Image.fromarray(np.uint8(combined_image_mask))
                    image = Image.fromarray(np.uint8(original_image))
                    mask = Image.fromarray(mask)
                    image.save(f'{image_save_path}count_{count_label_regioms}_{original_image_name}')
                    mask.save(f'{masks_save_path}count_{count_label_regioms}_{original_image_name}')
                    combined_image_mask.save(combined_save_path+original_image_name)

                    df = df.append({'filename':f"count_{count_label_regioms}_{original_image_name}",'count':count_label_regioms, 
                                    'region locations':label_regions_cranberry_outline,'locations':label_regions_cranberry_outline_points,'dataset':dataset},ignore_index=True)
                except Exception as e:
                    print("file: {}, error: {}".format(mask_link,e))
            elif instance_seg:
                if multi_channel_masks:
                    single_mask = np.zeros((image_height,image_width,1),dtype=np.int8)
                    boxes = []
                    for list in label_regions_cranberry_outline:
                        x = [dict['x'] for dict in list]
                        y = [dict['y'] for dict in list]
                        bbox = bounding_box(x,y)
                        boxes.append(bbox)
                        print(bbox)
                        x,y = np.array(x),np.array(y)
                        xx,yy = polygon(y,x)
                        single_mask[xx,yy] = 255
                        combined_image_mask = np.dstack((combined_image_mask,single_mask))
                    df = df.append({'filename':original_image_name,'count':count_label_regioms,'boxes':boxes,'dataset':dataset,'colors':255},ignore_index=True)
                    
                elif single_channel_mask:
                    # single_mask = np.zeros((image_height,image_width,3),dtype=np.int8)
                    single_mask = np.zeros((image_height,image_width),dtype=np.int8)
                    color_number_list = []
                    boxes = []
                    color_counter = 1
                    for list in label_regions_cranberry_outline:
                        color_number = random.sample(range(10,255),3)
                        if color_number not in color_number_list:
                            color_number_list.append(color_number)
                        else:
                            while color_number not in color_number_list:
                                color_number = random.sample(range(10,255),3)
                            color_number_list.append(color_number)
                        x = [dict['x'] for dict in list]
                        y = [dict['y'] for dict in list]
                        bbox = bounding_box(x,y)
                        boxes.append(bbox)
                        # print(bbox)
                        x,y = np.array(x),np.array(y)
                        xx,yy = polygon(y,x)
                        # single_mask[xx,yy,:] = color_number
                        single_mask[xx,yy] = color_counter
                        color_counter = color_counter + 1
                    df = df.append({'filename':original_image_name,'count':count_label_regioms,'boxes':boxes,'dataset':dataset,'colors':color_number_list},ignore_index=True)
                    
                    
                    
                    
                    
                    
                    combined_image_mask = np.dstack((combined_image_mask,single_mask))
                # plt.imshow(single_mask)
                # plt.show()
            
                combined_save_path = save_path+"instance_seg/"
                if single_channel_mask:
                    combined_mask_save_path = combined_save_path + "combined_single_mask/"
                    image_save_path = combined_save_path + "images_multicolor/"
                    masks_save_path = combined_save_path + "masks_multicolor/"
                    utils.create_dir_if_doesnt_exist(image_save_path)
                    utils.create_dir_if_doesnt_exist(masks_save_path)
                    image = Image.fromarray(np.uint8(original_image))
                    mask = Image.fromarray(np.uint8(single_mask))
                    # print(mask.size)
                    image.save(image_save_path+original_image_name)
                    mask.save(masks_save_path+original_image_name)
                else:
                    combined_mask_save_path = combined_save_path + "combined_multi_mask/"
                utils.create_dir_if_doesnt_exist(combined_mask_save_path)
                np.save(combined_mask_save_path+original_image_name+".npy",combined_image_mask)
            # df = df.append({'filename':"combined_mask_"+original_image_name,'count':count_label_regioms, 'locations':label_regions_cranberry_outline,
            #                 'dataset':dataset},ignore_index=True)
        
    
    if points:
        df.to_csv(save_path+"points/gt.csv")
    else:
        df.to_csv(save_path+"semantic_seg/gt.csv")
    # print(df)
    
        


latest_seg_path = root_path + "/data/cranberry/jsons/export-2020-02-03T15_16_10.598Z.json"
latest_point_path = root_path + "/data/cranberry/jsons/export-2020-02-11T16_07_10.855Z.json"
points = False
if points:
    images_save_path = "/data/cranberry/"
    path = latest_point_path
    utils.create_dir_if_doesnt_exist(root_path + images_save_path)
    
else:
    images_save_path = "/data/cranberry/"
    path = latest_seg_path
    utils.create_dir_if_doesnt_exist(root_path + images_save_path)

# color_number = random.sample(range(20,255),3)
# print(color_number)
labelbox_json_to_csv(path,points=points,save_images=True,save_path=root_path + images_save_path)

# import matplotlib.pyplot as plt
# image_path = "/home/native/Downloads/panoptic_annotations_trainval2017/annotations/panoptic_val2017/panoptic_val2017/000000000872.png"
# image = Image.open(image_path)
# print(image.size)
# image = np.asarray(image)
# print(image.shape)
# plt.imshow(image)
# plt.show()
# img = image[:,:,:3]
# mask = image[:,:,3]
# plt.imshow(mask)
# plt.show()
# image.show()
