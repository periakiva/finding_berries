import sys
import os
current_path = os.getcwd().split("/")
if 'projects' in current_path:
    sys.path.append("/home/native/projects/finding_berries/")
else:
    sys.path.append("/data/finding_berries/")

import gc
import comet_ml
import utils.utils as utils
import torch
from scipy import ndimage
import torch.optim as optim
from torch import nn
import torchvision as tv
from torchvision.models import inception_v3
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from models import unet, loss, unet_refined
from models.unet_regression import unet_regres
from peterpy import peter
from datasets.cranberries import cranberry_dataset
import numpy as np
from tqdm import tqdm
import utils.eval_utils as eval_utils
import datetime
from skimage.morphology import watershed
from skimage.segmentation import find_boundaries
from skimage import morphology
from matplotlib import colors
import warnings
import yaml
warnings.filterwarnings('ignore')

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


class Evaluator(object):
    # def __init__(self,model,test_loader,criterion,has_mask=False):
    def __init__(self,model,test_loader,criterion,
                test_with_full_supervision = 0,has_mask=True):

        self.model = model
        self.criterion = criterion
        self.save_figures = False
        self.visualizer_indicator = False
        if test_loader is not None:
            self.test_loader = test_loader
            self.test_with_full_supervision = test_with_full_supervision


    def visualizer(self,pred,imgs,masks,estimated_count,detection_count,gt_count):

        if pred.shape[0]>1 and len(pred.shape)==3:
            print(f"pred mask: {pred.shape}")
            pred = pred[0,:,:]
            imgs = imgs[0,:,:,].unsqueeze_(0)
            masks = masks[0,:,:,].unsqueeze_(0)
        imgs = imgs.cpu().detach().numpy().squeeze()
        masks = masks.cpu().detach().numpy()
        gt = np.transpose(masks,(1,2,0)).squeeze()
        image = np.transpose(imgs,(1,2,0))

        blobs = pred==1
        gt = gt == 1
        
        # labels, nlabels = ndimage.label(blobs)
        labels, nlabels = morphology.label(blobs,return_num=True)
        gt_labels,ngt = morphology.label(gt,return_num=True)

        file_name = f"gt_{gt_count}_pred_{nlabels}"

        cmap = plt.cm.get_cmap('tab10')
        cmap = rand_cmap(100,type='bright',first_color_black=False,last_color_black=False,verbose=False)

        labels_imshow = np.ma.masked_where(labels==0,labels)
        gt_imshow = np.ma.masked_where(gt==0,gt)
        gt_labels_imshow = np.ma.masked_where(gt_labels==0,gt_labels)
        edges = find_boundaries(labels,mode='outer')
        gt_edges = find_boundaries(gt_labels,mode='inner')
        gt_edges_imshow = np.ma.masked_where(gt_edges==0,gt_edges)
        edges_imshow = np.ma.masked_where(edges==0,edges)
        edges_cmap = colors.ListedColormap(['Cyan'])
        gt_cmap = colors.ListedColormap(['Black'])


        instance_fig = plt.figure()

        ax1 = instance_fig.add_subplot(1,1,1)
        # ax1.title.set_text("Semantic Prediction")
        ax1.imshow(image)
        ax1.imshow(labels_imshow,cmap=cmap,alpha=0.8)
        ax1.imshow(edges_imshow,cmap=edges_cmap)
        # plt.show()
        ax1.set_axis_off()
        plt.axis('off')
        # plt.margins(0,0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        # instance_fig.savefig(f"/home/native/projects/data/cranberry/visuals/paper/visual_results/ours/pred_{file_name}.png",dpi=600,bbox_inches='tight',pad_inches = 0)
        

        gt_fig = plt.figure()
        ax2 = gt_fig.add_subplot(1,1,1)
        # ax2.title.set_text("GT")
        ax2.imshow(image)
        ax2.imshow(gt_labels_imshow,cmap = cmap,alpha=0.8)
        ax2.imshow(gt_edges_imshow,cmap=edges_cmap)
        # plt.show()
        ax2.set_axis_off()
        plt.axis('off')
        # plt.margins(0,0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        gt_fig.savefig(f"/home/native/projects/data/cranberry/visuals/paper/visual_results/overlaid_ground_truth/overlaid_gt_{file_name}.png",dpi=600,bbox_inches='tight',pad_inches = 0)

        image_fig = plt.figure()
        ax3 = image_fig.add_subplot(1,1,1)
        # ax3.title.set_text("Image")
        ax3.imshow(image)
        ax3.set_axis_off()
        plt.axis('off')
        # plt.margins(0,0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())

        instance_fig.clear()
        gt_fig.clear()
        image_fig.clear()
        plt.cla()
        plt.clf()
        plt.close('all')
        plt.close(instance_fig)
        plt.close(gt_fig)
        plt.close(image_fig)
        gc.collect()
        return


    def evaluate(self,cometml_experiment):
        self.model.eval()
        print("testing")
        total_loss = 0
        gt_pred_count = {}
        preds,targets,counts,estimated_counts,detection_counts = [],[],[],[], []
        # loss_type = "_".join(self.losses_to_use)
        # loss_weights_str = "_".join([str(x)+str(y) for x,y in self.loss_weights.items()])
        if self.test_with_full_supervision == 1:
            loader = self.test_loader
        with torch.no_grad():
            for batch_index,batch in enumerate(loader):
                imgs,masks,count,img_path = batch

                imgs = imgs.to(device)
                masks = masks.to(device).squeeze(1)
                count = count.to(device)
                output, count_estimation = self.model.forward(imgs)
                
                # loss = self.criterion(output,masks)
                pred = output.max(1)[1].squeeze_(1).squeeze_(0).cpu().numpy()
                
                blobs = pred==1
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



                if self.visualizer_indicator:
                        self.visualizer(pred,imgs,masks,count_estimation.item(),count_by_detection,count.item())
                        # cometml_experiment.log_figure(figure_name=f"test, current loss: {loss}",figure=figure)
                        # model.predict(batch)
                        if self.save_figures:
                            # print("saving image")
                            figure_save_path = f"/home/native/projects/cranberry_counting/visualization_outputs/points/{loss_type}_{loss_weights_str}/"
                            utils.create_dir_if_doesnt_exist(figure_save_path)
                            figure.savefig(f"{figure_save_path}/testing_loss_{loss}_detectcount_{count_by_detection}_estimatedcount_{count_estimation.item()}_gt_count_{count.item()}.png",dpi=300)
                        # figure.clear()
                        # plt.cla()
                        # plt.clf()
                        # plt.close('all')
                        # plt.close(figure)
                        # gc.collect()
                gt_pred_count[img_path] = (count_by_detection,count.item())
                masks = masks.squeeze_(0).cpu().numpy()
                preds.append(pred)
                targets.append(masks)
                counts.append(count.item())
                estimated_counts.append(count_estimation.item())
                detection_counts.append(count_by_detection)
                # total_loss+=loss.item()
        # val_mae_lcfcn = eval_utils.val_mae(estimated_counts,counts)
        count_mae = eval_utils.mae(estimated_counts,counts)
        count_rmse = eval_utils.rmse(estimated_counts,counts)
        count_mape = eval_utils.mape(estimated_counts,counts)

        detection_count_mae = eval_utils.mae(detection_counts,counts)
        detection_count_rmse = eval_utils.rmse(detection_counts,counts)
        detection_count_mape = eval_utils.mape(detection_counts,counts)
        count_metrics = {"regression mae":count_mae,"regression rmse":count_rmse,"regression mape":
                        count_mape,"detection mae":detection_count_mae,"detection rmse":detection_count_rmse,
                        "detection mape":detection_count_mape}
        # print(type(count_metrics[0]))
        _,_,mean_iou,_ = eval_utils.calc_mAP(preds,targets)
        print("Validation mIoU value: {0:1.5f}".format(mean_iou))
        print(f"Validation Count Regression Mean Average Error: {count_mae}\nRegression Root Mean Squared Error: {count_rmse}\nRegression Mean Absolute Percent Error: {count_mape}\nDetection MAE: {detection_count_mae}\nDetection RMSE: {detection_count_rmse}\n Detection MAPE: {detection_count_mape}")
        # print("Validation average loss: {1:1.2f}".format(total_loss/self.val_loader.__len__()))
        with open("ours_count_pred.yaml",'w') as file:
            yaml.dump(gt_pred_count,file)
        return
        # return total_loss/self.val_loader.__len__(), mean_iou,count_metrics

    
    def forward(self,cometml_experiment):
        with cometml_experiment.validate():
            self.evaluate(cometml_experiment)
        return

if __name__ == "__main__":

    project_name = f"{current_path[-3]}_{current_path[-1]}"#_{datetime.datetime.today().strftime('%Y-%m-%d-%H:%M')}"
    experiment = comet_ml.Experiment(api_key="9GTK1r9PK4NzMAoLsnC6XxI7p",project_name=project_name,workspace="periakiva")
    
    config_path = utils.dictionary_contents(os.getcwd()+"/",types=["*.yaml"])[0]
    config = utils.config_parser(config_path,experiment_type="training")

    torch.set_default_dtype(torch.float32)
    device_cpu = torch.device('cpu')
    device = torch.device('cuda:0') if config['use_cuda'] else device_cpu

    # data_dictionary,batch_size,num_workers,instance_seg = False):
    test_loader = cranberry_dataset.build_single_loader(data_dictionary = config['data']['eval_dir'],
                                                        batch_size=config['testing']['batch_size'],
                                                        num_workers=config['testing']['num_workers'],
                                                        type=config['data']['type'], has_mask = config['data']['has_mask']
                                                        )


    with peter('Building Network'):
        model = unet_refined.UNetRefined(n_channels=3,n_classes=2)
        # model = unet_regres.Unet(in_channels=3,classes=2,decoder_channels= (512,256,128),encoder_depth=3)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("model has {} trainable parameters".format(num_params))
    # model = nn.DataParallel(model)
    model.to(device)
    model.cuda()
    

    class_weights = torch.Tensor((1,1)).float()
    class_weights = class_weights.to(device)
    loss_segmentation = nn.CrossEntropyLoss(class_weights)
    # loss_convexity = loss.ConvexShapeLoss(height=456,width=608,device=device)
    optimizer = optim.Adam(model.parameters(),
                            lr=config['testing']['learning_rate'],
                            amsgrad=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,len(test_loader),eta_min = config['testing']['learning_rate'])
    start_epoch = 0
    lowest_mahd = np.infty
    #TODO: Add resume option to Trainer using below code
    if config['testing']['resume'] != False:
        with peter('Loading checkpoints'):
            if os.path.isfile(config['testing']['resume']):
                # model = torch.load(config['training']['resume'])
                checkpoint = torch.load(config['testing']['resume'])
                # print(checkpoint)
                start_epoch = checkpoint['epoch']
                model.load_state_dict(checkpoint['model'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                # scheduler.load_state_dict(checkpoint['scheduler'])
                print(f"loaded model from {config['testing']['resume']}")
                # print("Loaded checkpoint {}, now at epoch: {}".format(config['training']['resume'],checkpoint['epoch']))        
            else:
                print("no checkpoint found at {}".format(config['testing']['resume']))
                exit()


    
    class_weights = torch.Tensor((1,1)).float()
    class_weights = class_weights.to(device)
    loss_segmentation = nn.CrossEntropyLoss(class_weights)


    evalutor = Evaluator(model=model,test_loader = test_loader,
                        criterion=loss_segmentation,has_mask=config['data']['has_mask'],
                        test_with_full_supervision = config['testing']['test_with_full_supervision'])
    evalutor.forward(experiment)