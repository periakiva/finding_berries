import sys
import os
current_path = os.getcwd().split("/")
if 'projects' in current_path:
    sys.path.append("/home/native/projects/cranberry_counting/")
else:
    sys.path.append("/data/cranberry_counting/")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import matplotlib as mpl
from scipy import ndimage

if __name__ == '__main__':
    file_name = "108_2019_10_02_4864_1824_3040.png"
    img_path = f'/home/native/projects/data/cranberry/instance_seg/images_multicolor/{file_name}'
    mask_path = f'/home/native/projects/data/cranberry/instance_seg/masks_multicolor/{file_name}'

    fig = plt.figure()
    # cmap = plt.cm.get_cmap('tab20c')

    n = 10
    from_list = mpl.colors.LinearSegmentedColormap.from_list
    cmap = from_list(None, plt.cm.Set1(range(0,n)), n)

    img = np.asarray(Image.open(img_path))
    mask = np.asarray(Image.open(mask_path))
    print(f"img: {img.shape}, mask: {mask.shape}")
    labels_imshow = np.ma.masked_where(mask==0,mask)
    struct = ndimage.generate_binary_structure(2, 2)
    erode = ndimage.binary_erosion(mask, struct)
    edges = mask ^ erode
    plt.imshow(edges)
    plt.show()
    # ax = fig.add_subplot(1,1,1)
    # ax.imshow(img)
    # ax.imshow(labels_imshow,cmap=cmap,alpha=0.75,vmin=0)
    # plt.show()
    # fig.savefig(f"/home/native/projects/data/cranberry/visuals/paper/dataset_examples/{file_name}",dpi=300)
    
