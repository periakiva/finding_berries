import os
import finding_berries.utils.utils as utils
import torch
from scipy import ndimage
import torch.optim as optim
from finding_berries.models import unet_refined
from peterpy import peter
import numpy as np
import warnings
import torchvision
import argparse
from PIL import Image
from skimage import morphology
from tqdm import tqdm
from skimage.segmentation import find_boundaries

import matplotlib.pyplot as plt
from matplotlib import colors

warnings.filterwarnings('ignore')
current_path = os.getcwd().split("/")

class SingleEvaluator(object):
    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model

    def count_from_prediction(self, prediction: torch.Tensor) -> torch.Tensor:
        
        blobs = prediction==1
        labels, nlabels = ndimage.label(blobs)
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


    def evaluate_single(self, images, save=None):
        
        self.model.eval()
        print("testing")
        
        with torch.no_grad():
            pbar = tqdm(images, total=len(images))
            for image_path in pbar:
                image_name = image_path.split('/')[-1].split('.')[0]
                image = Image.open(image_path).convert("RGB")
                image = test_transform(image).unsqueeze_(0)
                image = image.to(device)
                output, count_estimation = self.model.forward(image)
                
                pred = output.max(1)[1].squeeze_(1).squeeze_(0).cpu().numpy()
                
                count_by_detection = self.count_from_prediction(pred)
                
                image_show = np.transpose(image.cpu().detach().numpy().squeeze(),(1,2,0))
                blobs = pred==1
                labels, nlabels = morphology.label(blobs, return_num=True)
                boundaries = find_boundaries(pred, mode='thick')
                boundaries_imshow = np.ma.masked_where(boundaries==0, boundaries)
                
                labels_imshow = np.ma.masked_where(labels==0,labels)
                
                cmap = plt.cm.get_cmap('tab10')
                edges_cmap = colors.ListedColormap(['Cyan'])
                
                figure = plt.figure(figsize=(30, 30), dpi=300)
                ax1 = figure.add_subplot(1,3,1)
                ax2 = figure.add_subplot(1,3,2)
                ax3 = figure.add_subplot(1,3,3)
                
                ax1.imshow(image_show)
                ax1.title.set_text("Input Image")
                
                ax2.imshow(pred)
                ax2.imshow(boundaries_imshow, cmap=edges_cmap)
                ax2.title.set_text("Prediction")
                
                ax3.imshow(image_show)
                ax3.imshow(labels_imshow, cmap=cmap, alpha=0.95)
                ax3.imshow(boundaries_imshow, cmap=edges_cmap)
                ax3.title.set_text("Overlayed Instance Predictions")
                
                figure.suptitle(f"Application of Triple-S on User Defined Image.\nPredicted number of cranberries: {count_by_detection}")
                
                if save:

                    plots_path_save = f"{save}/"
                    fig_save_image_root = (f"{save}/image/", ax1)
                    fig_save_predictions_root = (f"{save}/predictions/", ax2)
                    fig_save_predictions_overlaid_root = (f"{save}/prediction_overlaid/", ax3)
                    
                    roots = [
                        fig_save_image_root,
                        fig_save_predictions_root,
                        fig_save_predictions_overlaid_root
                    ]
                    
                    figure.savefig(f"{save}/full_figures/{image_name}.png", bbox_inches='tight')
                    for root, ax in roots:
                        utils.create_dir_if_doesnt_exist(root)
                        file_path = f"{root}/{image_name}.png"
                        extent = ax.get_window_extent().transformed(figure.dpi_scale_trans.inverted())
                        figure.savefig(file_path, bbox_inches=extent)
                
                plt.show()
        return

if __name__ == "__main__":
    
    main_config_path = f"{os.getcwd()}/configs/segEval.yaml"
    config = utils.load_yaml_as_dict(main_config_path)
    
    parser = argparse.ArgumentParser(description='Getting image input')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the image for the forward pass')
    parser.add_argument('--save', type=str, default=None, help='Save figures to path')
    args = parser.parse_args()
    
    if os.path.isdir(args.image_path):
        images = utils.dictionary_contents(args.image_path, types=['*.png', '*.jpg', '*.jpeg'])
    else:
        images = [args.image_path]
    
    torch.set_default_dtype(torch.float32)
    device_cpu = torch.device('cpu')
    device = torch.device('cuda') if config['use_cuda'] else device_cpu
    
    test_transform = torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                    ])

    with peter('Building Network'):
        model = unet_refined.UNetRefined(n_channels=3,n_classes=2)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("model has {} trainable parameters".format(num_params))
    model.to(device)

    optimizer = optim.Adam(model.parameters(),
                            lr=config['testing']['learning_rate'],
                            amsgrad=True)
    
    start_epoch = 0
    lowest_mahd = np.infty
    #TODO: Add resume option to Trainer using below code
    if config['testing'][config['location']]['resume'] != False:
        with peter('Loading checkpoints'):
            if os.path.isfile(config['testing'][config['location']]['resume']):
                checkpoint = torch.load(config['testing'][config['location']]['resume'])
                start_epoch = checkpoint['epoch']
                model.load_state_dict(checkpoint['model'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                print(f"loaded model from {config['testing'][config['location']]['resume']}")
            else:
                print("no checkpoint found at {}".format(config['testing'][config['location']]['resume']))
                exit()

    evalutor = SingleEvaluator(model=model)
    evalutor.evaluate_single(images, save=args.save)
