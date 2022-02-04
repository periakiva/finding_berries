import sys
import os
current_path = os.getcwd().split("/")
if 'projects' in current_path:
    sys.path.append("/home/native/projects/cranberry_counting/")
else:
    sys.path.append("/data/cranberry_counting/")
import matplotlib; 
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import matplotlib as mpl
from scipy import ndimage
from skimage.segmentation import find_boundaries
from skimage import morphology
from matplotlib import colors
from random import shuffle

def rand_cmap(nlabels, type='bright', first_color_black=True, last_color_black=False, verbose=True):
    """
    Creates a random colormap to be used together with matplotlib. Useful for segmentation tasks
    :param nlabels: Number of labels (size of colormap)
    :param type: 'bright' for strong colors, 'soft' for pastel colors
    :param first_color_black: Option to use first color as black, True or False
    :param last_color_black: Option to use last color as black, True or False
    :param verbose: Prints the number of labels and shows the colormap. True or False
    :return: colormap for matplotlib
    """
    from matplotlib.colors import LinearSegmentedColormap
    import colorsys
    import numpy as np


    if type not in ('bright', 'soft'):
        print ('Please choose "bright" or "soft" for type')
        return

    if verbose:
        print('Number of labels: ' + str(nlabels))

    # Generate color map for bright colors, based on hsv
    if type == 'bright':
        randHSVcolors = [(np.random.uniform(low=0.0, high=1),
                          np.random.uniform(low=0.2, high=1),
                          np.random.uniform(low=0.9, high=1)) for i in range(nlabels)]

        # Convert HSV list to RGB
        randRGBcolors = []
        for HSVcolor in randHSVcolors:
            randRGBcolors.append(colorsys.hsv_to_rgb(HSVcolor[0], HSVcolor[1], HSVcolor[2]))

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]

        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    # Generate soft pastel colors, by limiting the RGB spectrum
    if type == 'soft':
        low = 0.6
        high = 0.95
        randRGBcolors = [(np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high)) for i in range(nlabels)]

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]
        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    # Display colorbar
    if verbose:
        from matplotlib import colors, colorbar
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(15, 0.5))

        bounds = np.linspace(0, nlabels, nlabels + 1)
        norm = colors.BoundaryNorm(bounds, nlabels)

        cb = colorbar.ColorbarBase(ax, cmap=random_colormap, norm=norm, spacing='proportional', ticks=None,
                                   boundaries=bounds, format='%1i', orientation=u'horizontal')

    return random_colormap


if __name__ == '__main__':
    file_names = ["270_2019_09_11_4864_912_00.png","270_2019_09_11_4864_00_608.png",
                    "270_2019_09_11_4864_912_1824.png","270_2019_09_11_4864_2736_1216.png",
                    "270_2019_09_11_4864_2736_2432.png","270_2019_09_11_4864_2280_3040.png"]
    for file_name in file_names:
    # file_name = "230_2019_10_18_4864_1368_1824.png
        img_path = f'/home/native/projects/data/cranberry/instance_seg/images_multicolor/{file_name}'
        mask_path = f'/home/native/projects/data/cranberry/instance_seg/masks_multicolor/{file_name}'

        fig = plt.figure()
        edges_figure = plt.figure()
        # cmap = plt.cm.get_cmap('tab20c')

        # cmap = plt.cm.get_cmap('tab10')
        # cmaplist = [cmap(i) for i in range(cmap.N)]
        # shuffle(cmaplist)
        # cmaplist = [cmaplist[0],cmaplist[4]]
        # cmap = matplotlib.colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, cmap.N)
        cmap = rand_cmap(50,type='bright',first_color_black=False,last_color_black=False,verbose=False)
        # n = 5
        # from_list = mpl.colors.LinearSegmentedColormap.from_list
        # cmap = from_list(None, plt.cm.Dark2(range(0,n)), n)
        edges_cmap = colors.ListedColormap(['Cyan'])
        img = np.asarray(Image.open(img_path))
        mask = np.asarray(Image.open(mask_path))
        labels, nlabels = morphology.label(mask,return_num=True)
        edges = find_boundaries(mask,mode='outer')
        print(f"img: {img.shape}, mask: {mask.shape}")
        labels_imshow = np.ma.masked_where(labels==0,labels)
        edges_imshow = np.ma.masked_where(edges==0,edges)
        
        # edges[edges==255] = 0 
        # ax_edges = edges_figure.add_subplot(1,1,1)
        # ax_edges.imshow(edges)
        # plt.imshow(edges)
        # plt.show()
        ax = fig.add_subplot(1,1,1)
        ax.imshow(img)
        
        ax.imshow(labels_imshow,cmap=cmap,alpha=0.60,vmin=0)
        ax.imshow(edges_imshow, cmap = edges_cmap,alpha=0.75)
        ax.set_axis_off()
        plt.axis('off')
        # plt.show()
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        fig.savefig(f"/home/native/projects/data/cranberry/visuals/paper/dataset_examples/tab10_shuffled_{file_name}",dpi=600,bbox_inches='tight',pad_inches = 0)
        # edges_figure.savefig(f"/home/native/projects/data/cranberry/visuals/paper/dataset_examples/endge.png",dpi=300)
    
