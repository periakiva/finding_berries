"""
engine.py
copied from https://github.com/pytorch/vision/blob/master/references/detection/engine.py
"""


import math
import sys
import time
import torch
import numpy as np
import torchvision.models.detection.mask_rcnn
import matplotlib.patches as patches
from coco_utils import get_coco_api_from_dataset
from coco_eval import CocoEvaluator
import utils
import cv2
import copy

def visualize_bboxes(images,targets):
    single_image = np.transpose(images[0].cpu().detach().numpy(),(1,2,0)).squeeze()
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    ax.imshow(single_image)
    for box in targets[0]['boxes']:
        # print(f"dict: {dict}")
        # box = dict['boxes']
        # print(f"box: {box}")
        # box = box.item()

        x1 = box[0].item()
        y1 = box[1].item()
        x2 = box[2].item()
        y2 = box[3].item()
        # print(f"x1:{x1} y1:{y1} x2:{x2} y2:{y2}")
        
        rect = patches.Rectangle((x1,y1),x2-x1,y2-y1,fill=False,edgecolor='r')
        ax.add_patch(rect)
        # cv2.rectangle(cvimg,(x1,y1),(x2,y2),(255,255,0))
    plt.show()

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)

        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        ts = copy.deepcopy(targets)
        # print(f"targets before model: {targets[0]['boxes']}")
        # print(f"n images: {len(images)}\nn boxes: {targets[0]['boxes'].shape}\nn labels: {targets[0]['labels'].shape}\nn masks: {targets[0]['masks'].shape}\n")
        loss_dict = model(images, targets)
        print(loss_dict)
        # print(f"targets after model: {targets[0]['boxes']}")
        losses = sum(loss for loss in loss_dict.values())
        # print(losses)
        # if losses.item() > 1:
        #     single_image = np.transpose(images[0].cpu().detach().numpy(),(1,2,0)).squeeze()
        #     fig = plt.figure()
        #     ax = fig.add_subplot(111, aspect='equal')
        #     ax.imshow(single_image)
        #     # print(np.unique(single_image))
        #     # cvimg = cv2.imread(img_path)
        #     # print(single_image.shape)
        #     # plt.imshow(single_image)
        #     # plt.show()
        #     # cvimg = np.uint8(single_image*255)
        #     # print(cvimg.shape)
        #     # cvimg = cvimg.astype(int)
            
        #     # r,g,b = cv2.split(cvimg)
        #     # cvimg = cv2.merge([b,g,r])
        #     # print(cvimg)
        #     # print(targets[0]['boxes'])
        #     # for box in ts[0]['boxes']:
        #     for box in targets[0]['boxes']:
        #         # print(f"dict: {dict}")
        #         # box = dict['boxes']
        #         # print(f"box: {box}")
        #         # box = box.item()

        #         x1 = box[0].item()
        #         y1 = box[1].item()
        #         x2 = box[2].item()
        #         y2 = box[3].item()
        #         # print(box)
        #         # print(f"x1:{x1} y1:{y1} x2:{x2} y2:{y2}")
                
        #         rect = patches.Rectangle((x1,y1),x2-x1,y2-y1,fill=False,edgecolor='r')
        #         ax.add_patch(rect)
                # cv2.rectangle(cvimg,(x1,y1),(x2,y2),(255,255,0))
            # plt.show()
            
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        # print(loss_dict_reduced)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        # print(losses_reduced)
        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            # visualize_bboxes(images,targets)
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


def visualizer(pred,imgs,masks,epoch):
    # print(pred.shape)
    # pred = pred[0,:,:,:]
    # print(masks[0])
    mask = masks[0,:,:]
    for index in range(masks.shape[0]):
        mask = torch.max(mask,masks[index,:,:])
    # pred = pred[:masks.shape[0],:,:]
    # masks = torch.max(torch.tensor([masks[i] for i in range(masks.shape[0])]))
    # print(mask)
    print(f"pred: {pred.shape} imgs: {imgs.shape}, masks: {masks.shape}")
    # if pred.shape[0]>1 and len(pred.shape)==3:
    #     pred = pred[0,:,:]
        # imgs = imgs[0,:,:,].unsqueeze_(0)
        # masks = masks[0,:,:,].unsqueeze_(0)

    fig = plt.figure()
    ax1 = fig.add_subplot(2,2,1)
    ax1.title.set_text("Image")
    ax1.imshow(np.transpose(imgs.cpu().detach().numpy().squeeze(),(1,2,0)))
    ax2 = fig.add_subplot(2,2,2)
    ax2.title.set_text("GT")
    ax2.imshow(mask.cpu().detach().numpy())
    # ax2.imshow(np.transpose(mask.cpu().detach().numpy(),(1,2,0)).squeeze())
    pred = torch.sum(pred,dim=0)
    pred = pred.cpu().detach().numpy().squeeze()
    print(pred.shape)
    ax3 = fig.add_subplot(2,2,3)
    ax3.title.set_text("Pred")
    ax3.imshow(pred)
    ax4 = fig.add_subplot(2,2,4)
    ax4.imshow(np.transpose(imgs.cpu().detach().numpy().squeeze(),(1,2,0)))
    ax4.imshow(pred,alpha=0.6)
    # fig.suptitle(f"Segmentation Results after {epoch} epochs",y=0.80)
    # counter = 3
    # for i in range(3,masks.shape[0]):
    #     if counter>25:
    #         break
    #     ax = fig.add_subplot(5,5,counter)
    #     ax.imshow(np.transpose(pred[i-3,:,:].cpu().detach().numpy(),(1,2,0)).squeeze())
    #     ax.title.set_text("Prediction")
    #     counter+=1
    # # ax3 = fig.add_subplot(1,masks.shape[0]+2,1)
    # # ax3.title.set_text("Prediction")
    # # ax3.imshow(np.transpose(pred.cpu().detach().numpy(),(1,2,0)).squeeze())
    # fig.suptitle(f"Segmentation Results after {epoch} epochs",y=0.80)
    return fig

import matplotlib.pyplot as plt
@torch.no_grad()
def evaluate(model, data_loader, device):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for index, (image, targets) in enumerate(metric_logger.log_every(data_loader, 100, header)):
        image = list(img.to(device) for img in image)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        ts = copy.deepcopy(targets)
        torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(image)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        # print(outputs)
        pred_mask = outputs[0]['masks']
        # print(pred_mask.shape)
        gt = ts[0]['masks']
        
        if index > 1:
            fig = visualizer(pred_mask,image[0],gt,epoch=0)
            plt.show()
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator