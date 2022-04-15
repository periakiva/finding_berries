import sys
import os
current_path = os.getcwd().split("/")
if 'projects' in current_path:
    sys.path.append("/home/native/projects/cranberry_counting/")
else:
    sys.path.append("/app/cranberry_counting/")

import gc
import comet_ml
import torch
from scipy import ndimage
import torch.optim as optim
from torch import nn
import torchvision as tv
from torchvision.models import inception_v3
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from peterpy import peter
import numpy as np
from tqdm import tqdm
import datetime
import yaml
# import losses


import finding_berries.utils.utils as utils
import torch
from scipy import ndimage
import torch.optim as optim
from torch import nn
import matplotlib.pyplot as plt
from finding_berries.models import unet, loss, unet_refined
from peterpy import peter
from finding_berries.datasets import cranberry_dataset
import numpy as np
from tqdm import tqdm
from finding_berries.datasets import build_dataset
import finding_berries.utils.eval_utils as eval_utils
from skimage.segmentation import find_boundaries
from skimage import morphology
from matplotlib import colors
import warnings
import yaml
import torchvision

warnings.filterwarnings('ignore')

class Evaluator(object):
    # def __init__(self,model,test_loader,criterion,has_mask=False):
    def __init__(self,model,test_loader,criterion,
                test_with_full_supervision = 0,has_mask=True):

        self.model = model
        self.criterion = criterion
        self.save_figures = False
        self.visualizer_indicator = True
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
        blobs = pred==1
        # labels, nlabels = ndimage.label(blobs)
        labels, nlabels = morphology.label(blobs,return_num=True)

        # count_by_detection = 0
        # for label in range(1,nlabels):
        #     inds = np.argwhere(labels==label)
        #     area = inds.shape[0]
        #     x = inds[:,0]
        #     y = inds[:,1]
        #     if area < 20:
        #         labels[x,y] = 0
        #     if area > 20:
        #         count_by_detection = count_by_detection + 1

        cmap = plt.cm.get_cmap('tab10')
        labels_imshow = np.ma.masked_where(labels==0,labels)

        fig = plt.figure()
        ax1 = fig.add_subplot(3,2,1)
        ax1.title.set_text("Semantic Prediction")
        ax1.imshow(pred)

        ax2 = fig.add_subplot(3,2,2)
        ax2.title.set_text("GT")
        ax2.imshow(np.transpose(masks,(1,2,0)).squeeze())

        ax3 = fig.add_subplot(3,2,3)
        ax3.title.set_text("Image")
        ax3.imshow(np.transpose(imgs,(1,2,0)))

        ax4 = fig.add_subplot(3,2,4)
        ax4.title.set_text("Instance Overlay")
        ax4.imshow(np.transpose(imgs,(1,2,0)))
        ax4.imshow(labels_imshow,interpolation='none',cmap=cmap,alpha=0.9,vmin=0)

        ax5 = fig.add_subplot(3,2,5)
        ax5.imshow(labels,cmap=cmap)
        ax5.title.set_text("Instance Prediction")
        
        fig.suptitle(f"gt count: {gt_count}, regress count: {round(estimated_count)} count_detection: {round(detection_count)}",
                        y=0.98)
        
        return fig


    def evaluate(self,cometml_experiment):
        self.model.eval()
        print("testing")
        total_loss = 0
        preds,targets,counts,estimated_counts,detection_counts = [],[],[],[], []
        # loss_type = "_".join(self.losses_to_use)
        # loss_weights_str = "_".join([str(x)+str(y) for x,y in self.loss_weights.items()])
        if self.test_with_full_supervision == 1:
            loader = self.test_loader
        with torch.no_grad():
            for batch_index,batch in enumerate(loader):
                imgs,masks,count = batch

                imgs = imgs.to(device)
                masks = masks.to(device).squeeze(1)
                count = count.to(device)
                output, count_estimation = self.model.forward(imgs)
                
                loss = self.criterion(output,masks)
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
                        figure = self.visualizer(pred,imgs,masks,count_estimation.item(),count_by_detection,count.item())
                        cometml_experiment.log_figure(figure_name=f"test, current loss: {loss}",figure=figure)
                        # model.predict(batch)
                        if self.save_figures:
                            # print("saving image")
                            figure_save_path = f"/home/native/projects/cranberry_counting/visualization_outputs/points/{loss_type}_{loss_weights_str}/"
                            utils.create_dir_if_doesnt_exist(figure_save_path)
                            figure.savefig(f"{figure_save_path}/testing_loss_{loss}_detectcount_{count_by_detection}_estimatedcount_{count_estimation.item()}_gt_count_{count.item()}.png",dpi=300)
                        figure.clear()
                        plt.cla()
                        plt.clf()
                        plt.close('all')
                        plt.close(figure)
                        gc.collect()
                masks = masks.squeeze_(0).cpu().numpy()
                preds.append(pred)
                targets.append(masks)
                counts.append(count.item())
                estimated_counts.append(count_estimation.item())
                detection_counts.append(count_by_detection)
                total_loss+=loss.item()
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
        print("Validation average loss: {1:1.2f}".format(total_loss/self.val_loader.__len__()))


        return total_loss/self.val_loader.__len__(), mean_iou,count_metrics

    
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
    # test_loader = cranberry_dataset.build_single_loader(data_dictionary = config['data']['eval_dir'],
    #                                                     batch_size=config['testing']['batch_size'],
    #                                                     num_workers=config['testing']['num_workers'],
    #                                                     instance_seg=config['data']['instance_seg'],
    #                                                     test=config['testing']['img_only'], has_mask = config['data']['has_mask']
    #                                                     )
    
    test_transform = torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                    # torchvision.transforms.Normalize(*mean_std)
                    ])
    
    test_loader = build_dataset(dataset_name='craid', 
                                root=config['data']['eval_dir'], 
                                batch_size=config['testing']['batch_size'],
                                num_workers=config['testing']['num_workers'], 
                                split="test", 
                                transforms=test_transform)

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
    model_paths = config['data']['model_name']
    for model_path in model_paths:
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

        # model = torch.load(config['data']['model_name'])
        # model.load_state_dict(checkpoint['model'])

        evalutor = Evaluator(model=model,test_loader = test_loader, criterion=loss_segmentation,has_mask=config['data']['has_mask'])
        evalutor.forward(experiment)