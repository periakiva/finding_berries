import numpy as np
from tqdm import tqdm as tqdm
import sys
import torch
import matplotlib.pyplot as plt


class Meter(object):
    '''Meters provide a way to keep track of important statistics in an online manner.
    This class is abstract, but provides a standard interface for all meters to follow.
    '''

    def reset(self):
        '''Resets the meter to default settings.'''
        pass

    def add(self, value):
        '''Log a new value to the meter
        Args:
            value: Next restult to include.
        '''
        pass

    def value(self):
        '''Get the value of the meter in the current state.'''
        pass


class AverageValueMeter(Meter):
    def __init__(self):
        super(AverageValueMeter, self).__init__()
        self.reset()
        self.val = 0

    def add(self, value, n=1):
        self.val = value
        self.sum += value
        self.var += value * value
        self.n += n

        if self.n == 0:
            self.mean, self.std = np.nan, np.nan
        elif self.n == 1:
            self.mean = 0.0 + self.sum  # This is to force a copy in torch/numpy
            self.std = np.inf
            self.mean_old = self.mean
            self.m_s = 0.0
        else:
            self.mean = self.mean_old + (value - n * self.mean_old) / float(self.n)
            self.m_s += (value - self.mean_old) * (value - self.mean)
            self.mean_old = self.mean
            self.std = np.sqrt(self.m_s / (self.n - 1.0))

    def value(self):
        return self.mean, self.std

    def reset(self):
        self.n = 0
        self.sum = 0.0
        self.var = 0.0
        self.val = 0.0
        self.mean = np.nan
        self.mean_old = 0.0
        self.m_s = 0.0
        self.std = np.nan

def visualizer(pred,imgs,masks,epoch):
        
        print(f"pred: {pred.shape} imgs: {imgs.shape}, masks: {masks.shape}")
        if pred.shape[0]>1 and len(pred.shape)==3:
            pred = pred[0,:,:]
            imgs = imgs[0,:,:,].unsqueeze_(0)
            masks = masks[0,:,:,].unsqueeze_(0)
        
        pred = pred.cpu().detach().numpy().squeeze()
        masks = masks.cpu().detach().numpy().squeeze()
        imgs = np.transpose(imgs.cpu().detach().numpy().squeeze(),(1,2,0))
        fig = plt.figure()
        ax1 = fig.add_subplot(1,4,1)
        ax1.title.set_text("Prediction")
        ax1.imshow(pred)
        ax2 = fig.add_subplot(1,4,2)
        ax2.title.set_text("GT")
        ax2.imshow(masks)
        ax3 = fig.add_subplot(1,4,3)
        ax3.title.set_text("Image")
        ax3.imshow(imgs)
        ax4 = fig.add_subplot(1,4,4)
        ax4.title.set_text("Overlay")
        ax4.imshow(imgs)
        ax4.imshow(pred,alpha=0.6)
        fig.suptitle(f"Segmentation Results after {epoch} epochs",y=0.80)
        return fig

class Epoch:

    def __init__(self, model, loss, metrics, stage_name, device='cpu', verbose=True):
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.stage_name = stage_name
        self.verbose = verbose
        self.device = device

        self._to_device()

    def _to_device(self):
        self.model.to(self.device)
        self.loss.to(self.device)
        for metric in self.metrics:
            metric.to(self.device)

    def _format_logs(self, logs):
        str_logs = ['{} - {:.4}'.format(k, v) for k, v in logs.items()]
        s = ', '.join(str_logs)
        return s

    def batch_update(self, x, y):
        raise NotImplementedError

    def on_epoch_start(self):
        pass

    def run(self, dataloader):

        self.on_epoch_start()

        logs = {}
        loss_meter = AverageValueMeter()
        metrics_meters = {metric.__name__: AverageValueMeter() for metric in self.metrics}

        with tqdm(dataloader, desc=self.stage_name, file=sys.stdout, disable=not (self.verbose)) as iterator:
            for x, y in iterator:
                x, y = x.to(self.device), y.to(self.device)
                loss, y_pred = self.batch_update(x, y)
                
                fig = visualizer(pred = y_pred,imgs = x,masks = y,epoch = 0)
                plt.show()

                loss_value = loss.cpu().detach().numpy()
                loss_meter.add(loss_value)
                loss_logs = {self.loss.__name__: loss_meter.mean}
                logs.update(loss_logs)

                # update metrics logs
                for metric_fn in self.metrics:
                    metric_value = metric_fn(y_pred, y).cpu().detach().numpy()
                    metrics_meters[metric_fn.__name__].add(metric_value)
                metrics_logs = {k: v.mean for k, v in metrics_meters.items()}
                logs.update(metrics_logs)

                if self.verbose:
                    s = self._format_logs(logs)
                    iterator.set_postfix_str(s)

        return logs


class TrainEpoch(Epoch):

    def __init__(self, model, loss, metrics, optimizer, device='cpu', verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name='train',
            device=device,
            verbose=verbose,
        )
        self.optimizer = optimizer

    def on_epoch_start(self):
        self.model.train()

    def batch_update(self, x, y):
        self.optimizer.zero_grad()
        prediction = self.model.forward(x)
        loss = self.loss(prediction, y)
        loss.backward()
        self.optimizer.step()
        return loss, prediction


class ValidEpoch(Epoch):

    def __init__(self, model, loss, metrics, device='cpu', verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name='valid',
            device=device,
            verbose=verbose,
        )

    def on_epoch_start(self):
        self.model.eval()

    def batch_update(self, x, y):
        with torch.no_grad():
            prediction = self.model.forward(x)
            loss = self.loss(prediction, y)
        return loss, prediction



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

        _,_,mean_iou,_ = eval_utils.calc_mAP(preds,targets)
        print("Validation mIoU value: {0:1.5f}".format(mean_iou))
        print("Validation Epoch {0:2d} average loss: {1:1.2f}".format(epoch+1, total_loss/self.val_loader.__len__()))
        cometml_experiemnt.log_metric("Validation mIoU",mean_iou)
        cometml_experiemnt.log_metric("Validation Average Loss",total_loss/self.val_loader.__len__())

        return total_loss/self.val_loader.__len__(), mean_iou
    
    def forward(self,cometml_experiment):
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
