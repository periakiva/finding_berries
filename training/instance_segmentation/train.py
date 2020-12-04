import sys
import os
current_path = os.getcwd().split("/")
if 'projects' in current_path:
    sys.path.append("/home/native/projects/cranberry_counting/")
else:
    sys.path.append("/app/cranberry_counting/")


import comet_ml
import utils.utils as utils
import torch
import torch.optim as optim
from torch import nn
import torchvision as tv
from torchvision.models import inception_v3
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from models import unet
from peterpy import peter
from datasets.cranberries import cranberry_dataset
import numpy as np
from tqdm import tqdm
import utils.eval_utils as eval_utils
import datetime
import warnings
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
        ax1 = fig.add_subplot(1,3,1)
        ax1.title.set_text("Prediction")
        ax1.imshow(pred)
        ax2 = fig.add_subplot(1,3,2)
        ax2.title.set_text("GT")
        ax2.imshow(np.transpose(masks.cpu().detach().numpy(),(1,2,0)).squeeze())
        ax3 = fig.add_subplot(1,3,3)
        ax3.title.set_text("Image")
        ax3.imshow(np.transpose(imgs.cpu().detach().numpy().squeeze(),(1,2,0)))
        fig.suptitle(f"Segmentation Results after {epoch} epochs",y=0.80)
        return fig

    def train(self,epoch,cometml_experiemnt):
        total_loss = 0 
        self.model.train()
        for batch_index,batch in enumerate(self.train_loader):
            imgs,masks = batch
            imgs = imgs.to(device)
            masks = masks.to(device).squeeze(1)
            self.optimizer.zero_grad()
            output = self.model.forward(imgs)
            loss = self.criterion(output,masks)
            loss.backward()
            self.optimizer.step()
            total_loss +=loss.item()
        cometml_experiemnt.log_metric("Training Average Loss",total_loss/self.train_loader.__len__())
        print("Training Epoch {0:2d} average loss: {1:1.2f}".format(epoch+1, total_loss/self.train_loader.__len__()))

        return total_loss/self.train_loader.__len__()
            
    def validate(self,epoch,cometml_experiemnt):
        self.model.eval()
        total_loss = 0
        preds,targets = [],[]
        with torch.no_grad():
            for batch_index,batch in enumerate(self.val_loader):
                imgs,masks = batch
                imgs = imgs.to(device)
                masks = masks.to(device).squeeze(1)

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

        _,_,mAP,_ = eval_utils.calc_mAP(preds,targets)
        print("Validation mAP value: {0:1.5f}".format(mAP))
        print("Validation Epoch {0:2d} average loss: {1:1.2f}".format(epoch+1, total_loss/self.val_loader.__len__()))
        cometml_experiemnt.log_metric("Validation mAP",mAP)
        cometml_experiemnt.log_metric("Validation Average Loss",total_loss/self.val_loader.__len__())

        return total_loss/self.val_loader.__len__(), mAP
    
    def forward(self,cometml_experiment):
        train_losses = []
        val_losses = []
        mAPs_val = []
        best_val_loss = np.infty
        best_val_mAP = 0
        for epoch in range(0,self.epochs):
            with cometml_experiment.train():
                train_loss = self.train(epoch,cometml_experiment)
            with cometml_experiment.validate():
                val_loss, val_mAP = self.validate(epoch,cometml_experiment)
            if val_loss < best_val_loss and val_mAP>best_val_mAP:
                best_val_loss = val_loss
                best_val_mAP = val_mAP
                model_save_dir = config.model_save_dir+f"{current_path[-1]}/{cometml_experiment.project_name}_{datetime.datetime.today().strftime('%Y-%m-%d-%H:%M')}/"
                utils.create_dir_if_doesnt_exist(model_save_dir)
                model_save_name = f"{current_path[-1]}_epoch_{epoch}_mAP_{val_mAP}_time_{datetime.datetime.today().strftime('%Y-%m-%d-%H:%M:%S')}.pth"
                torch.save(self.model,model_save_dir+model_save_name)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            mAPs_val.append(val_mAP)
        return train_losses, val_losses, mAPs_val

if __name__== "__main__":

    project_name = f"{current_path[-3]}_{current_path[-1]}"#_{datetime.datetime.today().strftime('%Y-%m-%d-%H:%M')}"
    experiment = comet_ml.Experiment(api_key="apikey",project_name=project_name,workspace="periakiva")
    
    config_path = utils.dictionary_contents(os.getcwd()+"/",types=["*.yaml"])[0]
    config = utils.config_parser(config_path,experiment_type="training")

    torch.set_default_dtype(torch.float32)
    device_cpu = torch.device('cpu')
    device = torch.device('cuda:0') if config['use_cuda'] else device_cpu

    train_dataloader, validation_dataloader, test_dataloader = cranberry_dataset.build_train_validation_loaders(
                                                                                                                data_dictionary=config.train_dir,batch_size=config.batch_size,
                                                                                                                num_workers=config.num_workers, instance_seg=True
                                                                                                                )

    with peter('Building Network'):
        model = unet.UNet(n_channels=3,n_classes=2)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("model has {} trainable parameters".format(num_params))
    model = nn.DataParallel(model)
    model.to(device)
    model.cuda()
    

    class_weights = torch.Tensor((1,1)).float()
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
    trainer = Trainer(model,train_dataloader,validation_dataloader,config.epochs,optimizer,loss_segmentation)
    train_losses, val_losses, mAPs_val = trainer.forward(experiment)
    # epoch = start_epoch
    # iteration = 0
    # train_losses = []
    # val_losses = []
    # mAPs_val = []
    # for epoch in range(0,config.epochs):
    #     train_loss = trainer.train(epoch)
    #     train_losses.append(train_loss)
    #     val_loss, val_mAP = trainer.validate(epoch)
    #     # train_loss.append(train(train_dataloader,epoch))
    #     # v_loss, v_mAP = validate(validation_dataloader,epoch)
    #     val_losses.append(v_loss)
    #     mAPs_val.append(val_mAP)


            


























