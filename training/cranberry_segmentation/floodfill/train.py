import sys
import os
current_path = os.getcwd().split("/")
if 'projects' in current_path:
    sys.path.append("/home/native/projects/finding_berries/")
else:
    sys.path.append("/data/cranberry_counting/")

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
import warnings
import yaml
import losses
warnings.filterwarnings('ignore')

class Trainer(object):
    def __init__(self,model,train_loader,val_loader,epochs,optimizer,scheduler,
                    criterion,losses_to_use,test_loader = None, test_with_full_supervision = 0,loss_weights=[1.0,1.0,1.0],class_weights={}):

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.class_weights = class_weights
        self.save_figures = False
        self.visualizer_indicator = True
        self.losses_to_use = losses_to_use
        self.loss_weights = loss_weights
        if len(self.losses_to_use) == 0:
            self.losses_to_use.append("none")
        if test_loader is not None:
            self.test_loader = test_loader
            self.test_with_full_supervision = test_with_full_supervision

    def visualizer(self,pred,imgs,masks,epoch,loss_type,estimated_count,gt_count):

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

        count_by_detection = 0
        for label in range(1,nlabels):
            inds = np.argwhere(labels==label)
            area = inds.shape[0]
            x = inds[:,0]
            y = inds[:,1]
            if area < 20:
                labels[x,y] = 0
            if area > 20:
                count_by_detection = count_by_detection + 1
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
        
        fig.suptitle(f"Segmentation with {loss_type} Loss Results after {epoch} epochs\ngt count: {gt_count}, regress count: {round(estimated_count)} count_detection: {round(count_by_detection)}",
                        y=0.98)
        
        return fig

    def train(self,epoch,cometml_experiemnt):
        total_loss = 0 
        # total_loss_dict = {"inst_loss""cvx_loss""circ_loss""closs"}
        self.model.train()
        losses_dict = {'seg_loss':0.0,'inst_loss':0.0,'cvx_loss':0.0,'circ_loss':0.0,'closs':0.0}
        for batch_index,batch in enumerate(self.train_loader):
            print(batch)
            imgs,masks,count = batch
            imgs = imgs.to(device)
            masks = masks.to(device).squeeze(1)
            self.optimizer.zero_grad()

            # output = self.model.forward(imgs)
            # S_numpy = utils.t2n(output[0])
            # masks_watershed = utils.t2n(masks).copy().squeeze()
            # masks_watershed[masks_watershed!=0] = np.arange(1,masks_watershed.sum()+1)
            # masks_watershed = masks_watershed.astype(float)
            # pred_mask = output.data.max(1)[1].squeeze().cpu().numpy()
            # probs = ndimage.black_tophat(pred_mask.copy(),7)
            # print(f"masks_watershed: {masks_watershed.shape} pred_mask: {pred_mask.shape}, probs: {probs.shape}")
            # seg = watershed(probs,masks_watershed)
            # s = find_boundaries(seg)

            # loss = self.criterion(output,masks)
            loss, loss_dict = losses.count_segment_loss(model,batch,self.losses_to_use,self.loss_weights,self.class_weights)
            loss.backward()
            self.optimizer.step()
            total_loss +=loss.item()
            for key in loss_dict.keys():
                losses_dict[key] +=loss_dict[key].item()
            
        if (epoch+1)%5 == 0:
            print(loss_dict)
        cometml_experiemnt.log_metric("Training Average Loss",total_loss/self.train_loader.__len__(),epoch=epoch+1)
        for key in losses_dict.keys():
            cometml_experiemnt.log_metric("Training " + key +" Loss",losses_dict[key]/self.train_loader.__len__(),epoch=epoch+1)
        # cometml_experiemnt.log_metrics(total_loss/self.train_loader.__len__(),epoch=epoch)
            
        print("Training Epoch {0:2d} average loss: {1:1.2f}".format(epoch+1, total_loss/self.train_loader.__len__()))

        return total_loss/self.train_loader.__len__()
            
    def validate(self,epoch,cometml_experiemnt):
        self.model.eval()
        print("validating")
        total_loss = 0
        preds,targets,counts,estimated_counts,detection_counts = [],[],[],[], []
        loss_type = "_".join(self.losses_to_use)
        loss_weights_str = "_".join([str(x)+str(y) for x,y in self.loss_weights.items()])
        if self.test_with_full_supervision == 1:
            loader = self.test_loader
        else:
            loader = self.val_loader
        with torch.no_grad():
            for batch_index,batch in enumerate(loader):
                imgs,masks,count = batch


                imgs = imgs.to(device)
                masks = masks.to(device).squeeze(1)
                count = count.to(device)
                output, count_estimation = self.model.forward(imgs)

                # output = self.model.forward(imgs)
                # count_estimation = torch.Tensor([0])
                
                loss = self.criterion(output,masks)
                pred = output.max(1)[1].squeeze_(1).squeeze_(0).cpu().numpy()
                
                blobs = pred==1
                labels, nlabels = ndimage.label(blobs)
                
                count_by_detection = utils.filterer(labels,nlabels)

                if self.visualizer_indicator:
                    if (epoch+1)%1 == 0:
                        figure = self.visualizer(pred,imgs,masks,epoch,loss_type,count_estimation.item(),count.item())
                        cometml_experiemnt.log_figure(figure_name=f"epoch: {epoch}, current loss: {loss}",figure=figure)
                        # model.predict(batch)
                        if self.save_figures:
                            # print("saving image")
                            figure_save_path = f"/home/native/projects/cranberry_counting/visualization_outputs/points/{loss_type}_{loss_weights_str}/"
                            utils.create_dir_if_doesnt_exist(figure_save_path)
                            figure.savefig(f"{figure_save_path}/epoch_{epoch}_loss_{loss}_estimatedcount_{count_estimation.item()}_gt_count_{count.item()}.png",dpi=300)
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
        print("Validation Epoch {0:2d} average loss: {1:1.2f}".format(epoch+1, total_loss/self.val_loader.__len__()))
        cometml_experiemnt.log_metric("Validation mIoU",mean_iou,epoch=epoch+1)
        cometml_experiemnt.log_metric("Validation Mean Average Error",count_mae,epoch=epoch+1)
        cometml_experiemnt.log_metric("Validation Root Mean Squared Error",count_rmse,epoch=epoch+1)
        cometml_experiemnt.log_metric("Validation Mean Absolute Percent Error",count_mape,epoch=epoch+1)
        cometml_experiemnt.log_metric("Validation Detection Mean Average Error",detection_count_mae,epoch=epoch+1)
        cometml_experiemnt.log_metric("Validation Detection Root Mean Squared Error",detection_count_rmse,epoch=epoch+1)
        cometml_experiemnt.log_metric("Validation Detection Mean Absolute Percent Error",detection_count_mape,epoch=epoch+1)
        cometml_experiemnt.log_metric("Validation Average Loss",total_loss/self.val_loader.__len__(),epoch=epoch+1)

        return total_loss/self.val_loader.__len__(), mean_iou,count_metrics
    
    def forward(self,cometml_experiment):
        train_losses = []
        val_losses = []
        mean_ious_val,mean_ious_val_list,count_metrics_list = [], [], []
        best_val_loss = np.infty
        # best_train_loss = np.infty
        best_val_mean_iou = 0
        best_mae = np.infty
        best_miou_to_mae_ratio = 0
        empty_string = "_"
        loss_weights_str = "_".join([str(x)+"_"+str(y) for x,y in self.loss_weights.items()])
        counting_type = self.losses_to_use[-1]
        model_save_dir = config['data']['model_save_dir']+f"{current_path[-1]}/{cometml_experiment.project_name}_{empty_string.join(self.losses_to_use)}_{loss_weights_str}_{datetime.datetime.today().strftime('%Y-%m-%d-%H:%M')}/"
        utils.create_dir_if_doesnt_exist(model_save_dir)
        for epoch in range(0,self.epochs):
            with cometml_experiment.train():
                train_loss = self.train(epoch,cometml_experiment)
            with cometml_experiment.validate():
                val_loss, val_mean_iou, count_metrics = self.validate(epoch,cometml_experiment)
            self.scheduler.step()
            val_mean_iou_list = val_mean_iou.tolist()
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            mean_ious_val.append(val_mean_iou)
            mean_ious_val_list.append(val_mean_iou_list)
            count_metrics_list.append(count_metrics)
            if val_mean_iou>best_val_mean_iou or best_mae > count_metrics["detection mae"] or best_mae > count_metrics["regression mae"]:
                best_val_loss = val_loss
                best_val_mean_iou = val_mean_iou
                if counting_type == "count_detect":
                    best_mae = count_metrics["detection mae"]
                elif counting_type == "count_regress":
                    best_mae = count_metrics["regression mae"]
                miou_to_mae_ratio = val_mean_iou*(1/best_mae)*100
                model_save_name = f"{current_path[-1]}_epoch_{epoch}_mean_iou_{val_mean_iou}_best_mae_{best_mae}_mioumao_ratio_{miou_to_mae_ratio}_time_{datetime.datetime.today().strftime('%Y-%m-%d-%H:%M:%S')}.pth"
                
                if best_miou_to_mae_ratio < miou_to_mae_ratio:
                    best_miou_to_mae_ratio = miou_to_mae_ratio
                        
                    with open(model_save_dir+"config.yaml",'w') as file:
                        print(mean_ious_val)
                        config['count_metrics'] = count_metrics_list
                        config['mean_ious_val'] = mean_ious_val_list
                        config['val_losses'] = val_losses
                        yaml.dump(config,file)
                    # torch.save(self.model,model_save_dir+model_save_name)
                    torch.save({'epoch': epoch, 'model': self.model.state_dict(), 
                                'optimizer': self.optimizer.state_dict(),
                                'loss':train_loss},model_save_dir+model_save_name)
            
        return train_losses, val_losses, mean_ious_val

if __name__== "__main__":

    project_name = f"{current_path[-3]}_{current_path[-1]}"#_{datetime.datetime.today().strftime('%Y-%m-%d-%H:%M')}"
    experiment = comet_ml.Experiment(api_key="9GTK1r9PK4NzMAoLsnC6XxI7p",project_name=project_name,workspace="periakiva")
    
    config_path = utils.dictionary_contents(os.getcwd()+"/",types=["*.yaml"])[0]
    config = utils.config_parser(config_path,experiment_type="training")
    # with open(config_path) as file:
    #     config = yaml.load(file,Loader=yaml.FullLoader)
    config['start_time'] = datetime.datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
    
    torch.set_default_dtype(torch.float32)
    device_cpu = torch.device('cpu')
    device = torch.device('cuda:0') if config['use_cuda'] else device_cpu
    # print(config['training']['train_val_test_split'][0])
    train_dataloader, validation_dataloader, test_dataloader = cranberry_dataset.build_train_validation_loaders(
                                                                                                                data_dictionary=config['data']['train_dir'],batch_size=config['training']['batch_size'],
                                                                                                                num_workers=config['training']['num_workers'],type=config['data']['type'],
                                                                                                                train_val_test_split=config['training']['train_val_test_split']
                                                                                                                )
    fs_test_loader = cranberry_dataset.build_single_loader(config['data']['test_dir'],batch_size=1,num_workers=1,test=True)
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
                            lr=config['training']['learning_rate'],
                            amsgrad=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,len(train_dataloader),eta_min = config['training']['learning_rate'])
    start_epoch = 0
    lowest_mahd = np.infty
    #TODO: Add resume option to Trainer using below code
    if config['training']['resume'] != False:
        with peter('Loading checkpoints'):
            if os.path.isfile(config['training']['resume']):
                # model = torch.load(config['training']['resume'])
                checkpoint = torch.load(config['training']['resume'])
                # print(checkpoint)
                start_epoch = checkpoint['epoch']
                model.load_state_dict(checkpoint['model'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                # scheduler.load_state_dict(checkpoint['scheduler'])
                print(f"loaded model from {config['training']['resume']}")
                # print("Loaded checkpoint {}, now at epoch: {}".format(config['training']['resume'],checkpoint['epoch']))        
            else:
                print("no checkpoint found at {}".format(config['training']['resume']))
                exit()
    print(f"using losses: {config['training']['losses_to_use']}")
    # running_average = utils.RunningAverage(len(train_dataloader))
    trainer = Trainer(model,train_dataloader,validation_dataloader,config['training']['epochs'],
                        optimizer,scheduler,loss_segmentation,losses_to_use=config['training']['losses_to_use'],
                        test_loader=fs_test_loader,
                        test_with_full_supervision=config['training']['test_with_full_supervision'],
                        loss_weights=config['training']['loss_weights'],class_weights = config['training']['class_weights'])
    train_losses, val_losses, mean_ious_val = trainer.forward(experiment)
    # epoch = start_epoch
    # iteration = 0
    # train_losses = []
    # val_losses = []
    # mean_ious_val = []
    # for epoch in range(0,config.epochs):
    #     train_loss = trainer.train(epoch)
    #     train_losses.append(train_loss)
    #     val_loss, val_mean_iou = trainer.validate(epoch)
    #     # train_loss.append(train(train_dataloader,epoch))
    #     # v_loss, v_mean_iou = validate(validation_dataloader,epoch)
    #     val_losses.append(v_loss)
    #     mean_ious_val.append(val_mean_iou)


            


























