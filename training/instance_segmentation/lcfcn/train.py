import sys
import os
current_path = os.getcwd().split("/")
if 'projects' in current_path:
    sys.path.append("/home/native/projects/cranberry_counting/")
else:
    sys.path.append("/data/cranberry_counting/")


import comet_ml
import utils.utils as utils
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
import torchvision as tv
from torchvision.models import inception_v3
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from models import unet, loss, unet_refined
from models.LCFCN import lcfcn
from peterpy import peter
from datasets.cranberries import cranberry_dataset
import numpy as np
from tqdm import tqdm
import utils.eval_utils as eval_utils
import datetime
import warnings
import yaml
import losses
warnings.filterwarnings('ignore')




class Trainer(object):
    def __init__(self,model,train_loader,val_loader,epochs,optimizer,criterion):

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.optimizer = optimizer
        self.criterion = criterion
        self.visualizer_indicator = True

    def visualizer(self,pred,imgs,masks,epoch):

        if pred.shape[0]>1 and len(pred.shape)==3:
            pred = pred[0,:,:]
            imgs = imgs[0,:,:,].unsqueeze_(0)
            masks = masks[0,:,:,].unsqueeze_(0)

        fig = plt.figure()
        ax1 = fig.add_subplot(1,4,1)
        ax1.title.set_text("Prediction")
        ax1.imshow(pred)
        ax2 = fig.add_subplot(1,4,2)
        ax2.title.set_text("GT")
        ax2.imshow(np.transpose(masks.cpu().detach().numpy(),(1,2,0)).squeeze())
        ax3 = fig.add_subplot(1,4,3)
        ax3.title.set_text("Image")
        ax3.imshow(np.transpose(imgs.cpu().detach().numpy().squeeze(),(1,2,0)))
        ax4 = fig.add_subplot(1,4,4)
        ax4.title.set_text("Overlay")
        ax4.imshow(np.transpose(imgs.cpu().detach().numpy().squeeze(),(1,2,0)))
        ax4.imshow(pred,alpha=0.6)
        fig.suptitle(f"Segmentation Results after {epoch} epochs",y=0.80)
        return fig



    def train(self,epoch,cometml_experiemnt=None):
        loss_sum = 0 
        self.model.train()
        train_dict = {}
        for batch_index,batch in enumerate(self.train_loader):
            # print(np.unique(batch['points'],return_counts=True))
            self.optimizer.zero_grad()
            loss = losses.lc_loss(model,batch)
            loss.backward()
            self.optimizer.step()
            loss_sum +=loss.item()
        cometml_experiemnt.log_metric("Training Average Loss",loss_sum/self.train_loader.__len__())
        print("Training Epoch {0:2d} average loss: {1:1.2f}".format(epoch+1, loss_sum/self.train_loader.__len__()))
        train_dict["loss"] = loss_sum / n_batches
        train_dict["epoch"] = epoch
        train_dict["n_samples"] = n_samples
        train_dict["iterations"] = n_batches
        return loss_sum/self.train_loader.__len__()
            
    def validate(self,epoch,cometml_experiemnt=None):
        self.model.eval()
        print("validating")
        total_loss = 0
        preds,targets = [],[]
        with torch.no_grad():
            for batch_index,batch in enumerate(self.val_loader):
                # imgs,masks = batch[]
                # imgs = imgs.to(device)
                # masks = masks.to(device).squeeze(1)
                output = self.model.forward(imgs)
                loss = self.criterion(output,masks)
                pred = output.max(1)[1].squeeze_(1).squeeze_(0).cpu().numpy()

                if self.visualizer_indicator:
                    if (epoch+1)%10 == 0:
                        figure = self.visualizer(pred,imgs,masks,epoch)
                        cometml_experiemnt.log_figure(figure_name=f"epoch: {epoch}, current loss: {loss}",figure=figure)
                masks = masks.squeeze_(0).cpu().numpy()
                preds.append(pred)
                targets.append(masks)
                total_loss+=loss.item()

        _,_,mean_iou,_ = eval_utils.calc_mAP(preds,targets)
        print("Validation mIoU value: {0:1.5f}".format(mean_iou))
        print("Validation Epoch {0:2d} average loss: {1:1.2f}".format(epoch+1, total_loss/self.val_loader.__len__()))
        cometml_experiemnt.log_metric("Validation mIoU",mean_iou)
        cometml_experiemnt.log_metric("Validation Average Loss",total_loss/self.val_loader.__len__())

        return total_loss/self.val_loader.__len__(), mean_iou
    
    def forward(self,cometml_experiment=None):
        train_losses = []
        val_losses = []
        mean_ious_val = []
        best_val_loss = np.infty
        best_val_mean_iou = 0
        model_save_dir = config['data']['model_save_dir']+f"{current_path[-1]}/{cometml_experiment.project_name}_{datetime.datetime.today().strftime('%Y-%m-%d-%H:%M')}/"
        utils.create_dir_if_doesnt_exist(model_save_dir)
        for epoch in range(0,self.epochs):
            with cometml_experiment.train():
                train_loss = self.train(epoch,cometml_experiment)
            with cometml_experiment.validate():
                val_loss, val_mean_iou = self.validate(epoch,cometml_experiment)
            if val_loss < best_val_loss and val_mean_iou>best_val_mean_iou:
                best_val_loss = val_loss
                best_val_mean_iou = val_mean_iou
                model_save_name = f"{current_path[-1]}_epoch_{epoch}_mean_iou_{val_mean_iou}_time_{datetime.datetime.today().strftime('%Y-%m-%d-%H:%M:%S')}.pth"
                # stream = file(model_save_dir+"config.yaml")
                with open(model_save_dir+"config.yaml",'w') as file:
                    yaml.dump(config,file)
                torch.save(self.model,model_save_dir+model_save_name)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            mean_ious_val.append(val_mean_iou)
        return train_losses, val_losses, mean_ious_val

if __name__== "__main__":

    project_name = f"{current_path[-3]}_{current_path[-1]}"#_{datetime.datetime.today().strftime('%Y-%m-%d-%H:%M')}"
    experiment = comet_ml.Experiment(api_key="9GTK1r9PK4NzMAoLsnC6XxI7p",project_name=project_name,workspace="periakiva")
    
    config_path = utils.dictionary_contents(os.getcwd()+"/",types=["*.yaml"])[0]
    config = utils.config_parser(config_path,experiment_type="training")
    # with open(config_path) as file:
    #     config = yaml.load(file,Loader=yaml.FullLoader)
    
    
    torch.set_default_dtype(torch.float32)
    device_cpu = torch.device('cpu')
    device = torch.device('cuda:0') if config['use_cuda'] else device_cpu
    # print(config['training']['train_val_test_split'][0])
    train_dataloader, validation_dataloader, test_dataloader = cranberry_dataset.build_train_validation_loaders(
                                                                                                                data_dictionary=config['data']['train_dir'],batch_size=config['training']['batch_size'],
                                                                                                                num_workers=config['training']['num_workers'],type=config['data']['type'],
                                                                                                                train_val_test_split=config['training']['train_val_test_split']
                                                                                                                )

    with peter('Building Network'):
        # model = unet_refined.UNetRefined(n_channels=3,n_classes=2)
        model = lcfcn.ResFCN(n_classes = 2)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("model has {} trainable parameters".format(num_params))
    # model = nn.DataParallel(model)
    model.to(device)
    model.cuda()
    

    class_weights = torch.Tensor((1,6000)).float()
    class_weights = class_weights.to(device)
    loss_segmentation = nn.CrossEntropyLoss(class_weights)
    # loss_convexity = loss.ConvexShapeLoss(height=456,width=608,device=device)
    optimizer = optim.Adam(model.parameters(),
                            lr=config['training']['learning_rate'],
                            amsgrad=True)
    start_epoch = 0
    lowest_mahd = np.infty
    #TODO: Add resume option to Trainer using below code
    if config['training']['resume'] != False:
        with peter('Loading checkpoints'):
            if os.path.isfile(config['training']['resume']):
                checkpoint = torch.load(config['training']['resume'])
                start_epoch = checkpoint['epoch']
                try:
                    lowest_mahd = checkpoint['mahd']
                except KeyError:
                    lowest_mahd = np.infty
                    print('Loaded checkpoint has not been validated')
                model.load_state_dict(checkpoint['model'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                print("Loaded checkpoint {}, now at epoch: {}".format(config['training']['resume'],checkpoint['epoch']))        
            else:
                print("no checkpoint found at {}".format(config['training']['resume']))
                exit()
    
    # running_average = utils.RunningAverage(len(train_dataloader))
    trainer = Trainer(model,train_dataloader,validation_dataloader,config['training']['epochs'],optimizer,loss_segmentation)
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


            


























