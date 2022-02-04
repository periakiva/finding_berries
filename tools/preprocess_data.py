import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import random
import numpy as np
import pickle
from pathlib import Path
from PIL import Image
import collections
import progressbar
import configargparse
import matplotlib.pyplot as plt
import utils.utils as utils
import PIL.ExifTags
from tqdm import tqdm
import pdb

def get_config(default_config_file):
    parser = configargparse.ArgParser(default_config_files=[default_config_file])
    # parser.add("-c","--my_config",required=True)
    parser.add("-m","--model_save_directory")
    parser.add("-r","--raw_images")
    parser.add("-l","--labels")
    parser.add("-c","--cropped_no_label")
    parser.add("-isp","--img_save_path")
    parser.add("-lsp","--label_save_path")
    parser.add("-hg","--height")
    parser.add("-wt","--width")
    parser.add("-rs","--root_sep")
    parser.add("-rc","--root_combined")
    parser.add("-name",required=True)
    config = parser.parse_args()
    return config


def all_files_in_root_to_one_list(root):
    img_paths = []
    for filename in Path(root).rglob("*.JPG"):
        if "Checkerboard" in str(filename) or "Artistic Shots" in str(filename):
            continue
        img_paths.append(filename)
    return img_paths

def crop_and_save(img_paths,height,width,directories,with_label=False):
    with progressbar.ProgressBar(max_value = len(img_paths)) as bar:
        for index,img_path in enumerate(img_paths):
            img_path = str(img_path)

            img = utils.load_image(img_path)
            date_time = utils.get_date_from_metadata(img)
            image_save_directory = directories[0] + date_time

            if not os.path.exists(image_save_directory):
                os.makedirs(image_save_directory)
            image = np.asarray(img)

            # if with_label:
                # json_path =args.main_jsons+ name+".json"
            #     label_points = utils.list_from_dict(dict_from_json(json_path)[0]['Label']['cranberry'])
                # mask = utils.mask_from_list(label_points,3648,4864)
                # img_mask_pair = join_img_label(img,mask)
            #   # label_name = "/"+str(index) + "_" + date_time + "_label"
            #     label_save_direcotry = directories[1] + date_time+"_label"
            #     cropped,cropped_label = utils.crop_and_save_single(image,height,width,with_label=True)
            # else:
            cropped = utils.crop_and_save_single(image,height,width,image_save_directory,str(index)+"_"+date_time,with_label=False)

            bar.update(1)

def main():
    args_path = "/home/native/projects/cranberry_counting/datasets/preprocess_images.yaml"
    config = get_config(args_path)
    img_paths = all_files_in_root_to_one_list(config.root_sep)
    print(len(img_paths))
    image_save_cropped_directory = config.img_save_path
    label_save_cropped_directory = config.label_save_path
    directories = (image_save_cropped_directory,label_save_cropped_directory)
    crop_and_save(img_paths,int(config.height),int(config.width),directories)

if __name__ == "__main__":
    main()
