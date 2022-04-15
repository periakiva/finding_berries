import os
import finding_berries.utils.utils as utils
import torch
from scipy import ndimage
import torch.optim as optim
from finding_berries.models import unet_refined
from peterpy import peter
import numpy as np
from finding_berries.datasets import build_dataset
import finding_berries.utils.eval_utils as eval_utils
import warnings
import torchvision

warnings.filterwarnings('ignore')
current_path = os.getcwd().split("/")

class Evaluator(object):
    def __init__(self, model: torch.nn.Module, test_loader: torch.utils.data.DataLoader) -> None:

        self.model = model
        self.test_loader = test_loader

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


    def evaluate(self):
        self.model.eval()
        print("testing")
        
        preds, targets, counts, estimated_counts, detection_counts = [], [], [], [], []
        
        with torch.no_grad():
            for batch_index, batch in enumerate(self.test_loader):
                imgs, masks, count, image_path = batch
                
                imgs = imgs.to(device)
                masks = masks.to(device).squeeze(1)
                count = count.to(device)
                output, count_estimation = self.model.forward(imgs)
                
                pred = output.max(1)[1].squeeze_(1).squeeze_(0).cpu().numpy()
                
                count_by_detection = self.count_from_prediction(pred)

                masks = masks.squeeze_(0).cpu().numpy()
                preds.append(pred)
                targets.append(masks)
                counts.append(count.item())
                estimated_counts.append(count_estimation.item())
                detection_counts.append(count_by_detection)
                
        count_mae = eval_utils.mae(estimated_counts,counts)
        count_rmse = eval_utils.rmse(estimated_counts,counts)
        count_mape = eval_utils.mape(estimated_counts,counts)

        detection_count_mae = eval_utils.mae(detection_counts,counts)
        detection_count_rmse = eval_utils.rmse(detection_counts,counts)
        detection_count_mape = eval_utils.mape(detection_counts,counts)
        
        count_metrics = {"regression mae":count_mae,"regression rmse":count_rmse,"regression mape":
                        count_mape,"detection mae":detection_count_mae,"detection rmse":detection_count_rmse,
                        "detection mape":detection_count_mape}
        
        _,_,mean_iou,_ = eval_utils.calc_mAP(preds, targets)
        print("Validation mIoU value: {0:1.5f}".format(mean_iou))
        print(f"Detection MAE: {detection_count_mae}\nDetection RMSE: {detection_count_rmse}\n Detection MAPE: {detection_count_mape}")
        return

    
    def forward(self):
        self.evaluate()
        return

if __name__ == "__main__":
    
    main_config_path = f"{os.getcwd()}/configs/segEval.yaml"
    config = utils.load_yaml_as_dict(main_config_path)
    project_name = f"{current_path[-3]}_{current_path[-1]}"#_{datetime.datetime.today().strftime('%Y-%m-%d-%H:%M')}"
    
    torch.set_default_dtype(torch.float32)
    device_cpu = torch.device('cpu')
    device = torch.device('cuda') if config['use_cuda'] else device_cpu
    
    test_transform = torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                    ])
    
    test_loader = build_dataset(dataset_name='craid', 
                                root=config['data'][config['location']]['eval_dir'], 
                                batch_size=config['testing']['batch_size'],
                                num_workers=config['testing']['num_workers'], 
                                split="test", 
                                transforms=test_transform)

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


    evalutor = Evaluator(model=model,
                         test_loader=test_loader)
    evalutor.forward()
