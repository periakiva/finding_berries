import sys
import os
current_path = os.getcwd().split("/")
if 'projects' in current_path:
    sys.path.append("/home/native/projects/cranberry_counting/")
else:
    sys.path.append("/data/cranberry_counting/")

print(sys.path)
import comet_ml
import random
import numpy as np
import csv
import finding_berries.utils.utils as utils
import torch
from finding_berries.datasets import cranberry_dataset
import pickle
import scipy.misc
import imageio
from PIL import Image
import matplotlib.pyplot as plt
from finding_berries.utils.utils import load_image,load_yaml_as_object,show_image,dictionary_contents,overlay_images, save_pickle
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

class svmTrainer(object):
    def __init__(self,kernel,train_loader,val_loader):
        self.type = type
        self.train_loader = train_dataloader
        self.val_loader = val_loader
        self.model = SVC(kernel=kernel,verbose=True)
        
    
    def train(self,cometml_experiemnt):
        for batch_index,batch in enumerate(self.train_loader):
            imgs,masks = batch
            print(imgs.shape)
            print(masks.shape)
            imgs = imgs.numpy()
            imgs = imgs.reshape(imgs.shape[0]*imgs.shape[2]*imgs.shape[3],imgs.shape[1])
            masks = masks.numpy()
            masks = masks.reshape(masks.shape[0]*masks.shape[2]*masks.shape[3],masks.shape[1])
            print(imgs.shape)
            print(masks.shape)
            self.model.fit(imgs,masks)
        print ('[INFO] Training Accuracy: %.2f' %self.model.score(imgs,masks))
        cometml_experiemnt.log_metric("Training Accuracy",self.model.score(imgs,masks))
    
    def validate(self):
        for batch_undex,batch in enumerate(self.val_loader):
            imgs,masks = batch
            imgs = imgs.numpy()
            imgs = imgs.reshape(imgs.shape[0]*imgs.shape[2]*imgs.shape[3],imgs.shape[1])
            masks = masks.numpy()
            masks = masks.reshape(masks.shape[0]*masks.shape[2]*masks.shape[3],masks.shape[1])
            predictions = self.model.predict(imgs)
            print(predictions==masks)


if __name__ == "__main__":

    project_name = f"{current_path[-1]}"#_{datetime.datetime.today().strftime('%Y-%m-%d-%H:%M')}"
    experiment = comet_ml.Experiment(api_key="apikey",project_name=project_name,workspace="periakiva")
    
    config_path = utils.dictionary_contents(os.getcwd()+"/",types=["*.yaml"])[0]
    config = utils.config_parser(config_path,experiment_type="training")
    
    torch.set_default_dtype(torch.float32)
    device_cpu = torch.device('cpu')
    device = torch.device('cuda:0') if config.use_cuda else device_cpu

    train_dataloader, validation_dataloader,test_dataloader = cranberry_dataset.build_train_validation_loaders(
                                                                                                                data_dictionary=config.train_dir,batch_size=config.batch_size,
                                                                                                                num_workers=config.num_workers
                                                                                                                )

    trainer = svmTrainer(kernel='linear',train_loader=train_dataloader,val_loader=validation_dataloader)
    trainer.train(experiment)
    trainer.validate()

    # for kernel in kernels:
    #     models[kernel] = train_model(X,y,kernel)
    #     model_name = "svm_"+kernel
    #     save_pickle(models[kernel],args.model_save_directory,model_name)