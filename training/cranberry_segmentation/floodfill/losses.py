import torch
import torch.nn.functional as F
import numpy as np 
from skimage.morphology import watershed, erosion
from skimage.segmentation import find_boundaries
from skimage.morphology import square
from skimage import morphology
from scipy import ndimage
import utils.utils as utils
from skimage import measure
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib
import pdb; 


def segmentation_loss(pred,gt,seg_weights):

    seg_loss = F.cross_entropy(pred, gt, weight=torch.Tensor((seg_weights[0],seg_weights[1])).float().cuda(),reduction=seg_weights[2])
    return seg_loss


def detection_based_count_loss(pred,gt_count):
    pred_mask = pred.data.max(1)[1].squeeze().cpu().numpy()
    blobs = pred_mask==1
    labels, nlabels = ndimage.label(blobs)
    # labels, nlabels = morphology.label(blobs)
    count = nlabels - 1
    
    closs = F.smooth_l1_loss(torch.Tensor([count]).cuda(),gt_count,reduction='mean')
    # print(f"closs: {closs}, gt: {gt_count}, predicted: {count}, tensor predicted: {torch.Tensor([count]).shape}, tensor predicted: {torch.Tensor([count])}")
    return closs

def regression_based_count_loss(count_estimation,gt_count):
    count_estimation_float = count_estimation#.float()
    gt_count_float = gt_count.float()
    
    closs = F.smooth_l1_loss(count_estimation_float,gt_count_float,reduction='mean')
    # print(f"closs: {closs}, gt: {gt_count_float}, predicted: {count_estimation_float}")
    return closs

def circularity_loss(pred,gt):
    """circularity_loss calculates and outputs the circularity loss

    Arguments:
        pred {tensor} -- prediction from model
        gt {tensor} -- ground truth tensor

    Returns:
        PyTorch huber loss -- circularity loss
    """
    pred_mask = pred.data.max(1)[1].squeeze().cpu().numpy()
    blobs = pred_mask==1
    labels, nlabels = ndimage.label(blobs)
    circularity_target = []
    circularity_pred = []
    for label in range(1,nlabels):
        inds = np.argwhere(labels==label)
        area = inds.shape[0]

        x = inds[:,0]
        y = inds[:,1]
        pts = [[x[i],y[i]] for i in range(len(x))]
        min_x = np.min(inds[:,0])
        min_y = np.min(inds[:,1])
        max_x = np.max(inds[:,0])
        max_y = np.max(inds[:,1])
        
        x_center = ((max_x - min_x)//2)+min_x
        y_center = ((max_y - min_y)//2)+min_y
        rect_area = (max_x-min_x+1)*(max_y-min_y+1)
        #TODO: change from rect area to convex hull area
        # r_min = (min_x - x_center)^2 + (min_y-y_center)^2
        # r_max = (max_x - x_center)^2 + (max_y - y_center)^2
        r_max = 0
        r_min = 1000
        rs = []
        for i,j in pts:
            r_tmp = abs((i-x_center)^2 + (j-y_center)^2)
            rs.append(r_tmp)
            if r_tmp < r_min:
                r_min = r_tmp
            if r_tmp > r_max:
                r_max = r_tmp
        assert r_max == max(rs)
        assert r_min == min(rs)
        delta_r = r_max - r_min
        if (rect_area <=20) or (max_x - min_x < 4) or (max_y - min_y < 4):
            circularity_measurement = 10.0
        else:
            circularity_measurement = delta_r
        circularity_pred.append(circularity_measurement)
        circularity_target.append(0.0)
        # print(f"circularity_measurement: {circularity_measurement}, delta_r: {delta_r}\nmin_x: {min_x}, min_y:{min_y}, max_x: {max_x}, max_y:{max_y}, rs: {rs}")
    
    circ_loss = F.smooth_l1_loss(torch.Tensor(circularity_pred),torch.Tensor(circularity_target))
    return circ_loss

def convexity_loss(pred,gt):
    """convexity_loss calculates and outputs the convexity loss

    Arguments:
        pred {tensor} -- prediction from model
        gt {tensor} -- ground truth

    Returns:
        PyTorch Huber loss -- regression loss over convexity
    """
    pred_mask = pred.data.max(1)[1].squeeze().cpu().numpy()
    blobs = pred_mask==1
    labels, nlabels = ndimage.label(blobs)
    convexity_pred = []
    convexity_target = []

    for label in range(1,nlabels):
        inds = np.argwhere(labels==label)
        area = inds.shape[0]

        x = inds[:,0]
        y = inds[:,1]
        pts = [[x[i],y[i]] for i in range(len(x))]
        min_x = np.min(inds[:,0])
        min_y = np.min(inds[:,1])
        max_x = np.max(inds[:,0])
        max_y = np.max(inds[:,1])
        rect_area = (max_x-min_x+1)*(max_y-min_y+1)
        #TODO: change from rect area to convex hull area
        polyarea = rect_area
        if (rect_area <=20) or (max_x - min_x < 4) or (max_y - min_y < 4):
            convexity_measurement = 0
        else:
            convexity_measurement = area/polyarea
        convexity_pred.append(convexity_measurement)
        convexity_target.append(1.0)

        # polyarea = 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

    cvx_loss = F.smooth_l1_loss(torch.Tensor(convexity_pred),torch.Tensor(convexity_target))
    if torch.isnan(cvx_loss):
        print(f"cvx loss: {cvx_loss}, labels: {labels}, nlabels: {nlabels}")
        # plt.imshow(utils.t2n(gt).squeeze())
        plt.imshow(pred_mask)
        
        plt.show()

    return cvx_loss

def instance_loss(pred,gt,background_points,instance_weights,imgs):
    """instance_loss Split loss/instance loss - the loss for making blobs instances

    Arguments:
        pred {tesnor} -- prediction from model
        gt {tensor} -- ground truth
        background_points {tensor} -- ground truth of points in backgrond
        instance_weights {dictionary} -- weights of instance loss
        imgs {tensor} -- input images

    Returns:
        PyTorch cross entropy loss -- split loss
    """
    # print(f"gt: {gt.shape}, background points: {background_points.shape}")
    points = utils.t2n(gt).copy().squeeze()
    image = np.transpose(utils.t2n(imgs).copy().squeeze(),(1,2,0))
    
    original_points = points.copy()
    background_points = utils.t2n(background_points).copy().squeeze()
    # print(np.unique(background_points))
    points_with_back = points+background_points

    # points[points>0] += 3
    # mode = "both"
    # mode = "selective"
    mode = "none"
    if mode == "selective":
        points, npoints = ndimage.label(points_with_back)
        label_to_ignore = []
        inds = np.argwhere(points>0)
        x_points = inds[:,0]
        y_points = inds[:,1]
        for x,y in zip(x_points,y_points):
            if background_points[x,y] == 2:
                label_to_ignore.append(points[x,y])
        labels_to_ignore = np.unique(label_to_ignore)

    elif mode == "none":
        points, npoints = ndimage.label(points)

    elif mode == "both":
        points, npoints = ndimage.label(points_with_back)
        label_to_ignore = []
        inds = np.argwhere(points>0)
        x_points = inds[:,0]
        y_points = inds[:,1]
        for x,y in zip(x_points,y_points):
            if background_points[x,y] == 2:
                label_to_ignore.append(points[x,y])
        labels_to_ignore = np.unique(label_to_ignore)
        edges_points,nedges_points = ndimage.label(original_points)
        edges_points = edges_points.astype(float)

    # points = points + 1
    points = points.astype(float)
    
    pred_mask = pred.data.max(1)[1].squeeze().cpu().numpy()
    # pred_softmax = F.softmax(pred,1)

    # distance_points = ndimage.distance_transform_edt(pred_mask)
    # points = points + 1



    # seg = watershed(pred_softmax,points,compactness=50.0,watershed_line=True)
    if mode == "selective":
        # points = points.reshape((points.shape[0],points.shape[1],1))
        # seg = watershed(image,points,compactness=0.01,watershed_line=True)
        # seg = np.amax(seg,axis=2)
        seg = watershed(pred_mask,points,compactness=0.05,watershed_line=True)
        for ignore_label in labels_to_ignore:
            seg[seg==ignore_label] = 0
        points_imshow = np.ma.masked_where(points_with_back!=1,points_with_back)
        background_points_imshow = np.ma.masked_where(points_with_back!=2,points_with_back)
        ws_inverse = seg.copy()
        ws_inverse[ws_inverse>0] = 1
    elif mode == "none":

        points = points.reshape((points.shape[0],points.shape[1],1))        
        seg = watershed(image,points,compactness=0.01)
        seg = np.amax(seg,axis=2)
        ws = find_boundaries(seg)
        ws_inverse = 1-ws
        boundaries_imshow = np.ma.masked_where(ws==0,ws)
    elif mode == "both":
        seg = watershed(pred_mask,points,compactness=0.05,watershed_line=True)
        for ignore_label in labels_to_ignore:
            seg[seg==ignore_label] = 0
        points_imshow = np.ma.masked_where(points_with_back!=1,points_with_back)
        background_points_imshow = np.ma.masked_where(points_with_back!=2,points_with_back)
        ws_inverse = seg.copy()
        ws_inverse[ws_inverse>0] = 1
        original_points = original_points.reshape((original_points.shape[0],original_points.shape[1],1))
        edges_seg = watershed(image,original_points,compactness=0.01)
        edges_seg = np.amax(edges_seg,axis=2)
        edges_ws = find_boundaries(edges_seg)
        edges_ws_inverse = 1-edges_ws
        boundaries_imshow = np.ma.masked_where(edges_ws==0,edges_ws)
    # seg[seg==labels_to_ignore] = 0

    
    eroded_ws_inverse = erosion(ws_inverse,square(6))
    # points_imshow = np.ma.masked_where(points==0,points)
    visualize = True
    if visualize:
        points_cmap = colors.ListedColormap(['blue'])
        back_points_cmap = colors.ListedColormap(['red'])
        edges_cmap = colors.ListedColormap(['blue'])
        cmap = plt.cm.get_cmap('tab20c')
        cmaplist = [cmap(i) for i in range(cmap.N)]
        
        cmaplist = [cmaplist[0],'white']
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, 2)
        

        if mode == "none":
            edges_fig = plt.figure()
            # ax1 = fig.add_subplot(2,2,1)
            # ax1.imshow(pred_mask)
            # ax2 = fig.add_subplot(2,2,2)
            # ax2.imshow(ws_inverse)
            # ax3 = fig.add_subplot(2,2,3)
            # ax3.imshow(seg)
            eroded_watershed = erosion(ws_inverse,square(6))
            ws_inverse_imshow = np.ma.masked_where(eroded_watershed==0,eroded_watershed)
            edges_cmap = colors.ListedColormap(['Cyan'])
            edges = find_boundaries(eroded_watershed,mode='inner')
            edges_imshow = np.ma.masked_where(edges==0,edges)

            ax5 = edges_fig.add_subplot(1,1,1)
            ax5.imshow(image)
            points = points.squeeze()
            points_imshow = np.ma.masked_where(points==0,points)
            ax5.imshow(ws_inverse_imshow,cmap =cmap, alpha = 0.5)
            ax5.imshow(edges_imshow,cmap=edges_cmap)
            ax5.imshow(points_imshow,cmap = points_cmap)
            # ax5.imshow(boundaries_imshow,cmap=edges_cmap)
            ax5.set_axis_off()
            edges_fig.savefig(f"/home/native/projects/data/cranberry/visuals/paper/dataset_examples/watershed/watershed_overlaid_c01_{npoints}",dpi=600,bbox_inches='tight')
            # plt.show()
            edges_fig.clear()
            plt.cla()
            plt.clf()
            plt.close('all')
            plt.close(edges_fig)
        elif mode == "selective":
            fig = plt.figure()
            ax4 = fig.add_subplot(1,1,1)
            edges_cmap = colors.ListedColormap(['Cyan'])
            edges = find_boundaries(eroded_ws_inverse,mode='inner')
            edges_imshow = np.ma.masked_where(edges==0,edges)
            eroded_ws_inverse_imshow = np.ma.masked_where(eroded_ws_inverse==0,eroded_ws_inverse)
            ax4.imshow(image)
            # ax4.imshow(eroded_ws_inverse,cmap = cmap)
            ax4.imshow(eroded_ws_inverse_imshow,cmap = cmap,alpha = 0.6)
            ax4.imshow(edges_imshow,cmap=edges_cmap)
            ax4.imshow(points_imshow,cmap=points_cmap)
            ax4.imshow(background_points_imshow,cmap=back_points_cmap)
            ax4.set_axis_off()
            plt.axis('off')
            # plt.show()
            fig.savefig(f"/home/native/projects/data/cranberry/visuals/paper/dataset_examples/selective_watershed/selective_watershed_overlaid_c01_{npoints}",dpi=600,bbox_inches='tight')
            fig.clear()
            plt.cla()
            plt.clf()
            plt.close('all')
            plt.close(fig)
        
        # fig.savefig(f"/home/native/projects/data/cranberry/visuals/paper/dataset_examples/selective_watershed_c05_{npoints}",dpi=600,bbox_inches='tight')
    
    return F.cross_entropy(pred,torch.LongTensor(eroded_ws_inverse).cuda()[None],weight=torch.Tensor((instance_weights[0],instance_weights[1])).float().cuda(),ignore_index=-100,reduction=instance_weights[2])
    

def count_segment_loss(model,batch,losses_to_use,loss_weights,class_weights):
    """count_segment_loss overall loss function for Triple-S model

    Arguments:
        model {nn.Module} -- model
        batch {batch} -- batch from dataloader iteration
        losses_to_use {list} -- list of losses to use - TODO: delete since it is not used anymore
        loss_weights {dictionary} -- dictionary with losses and weigts 
        class_weights {dictionary} -- dictionary with losses and weights for each class (here we have 2 classes) 

    Returns:
        loss -- overall loss of the model
    """
    model.train()

    imgs,masks,count = batch
    background_points = masks.clone()
    background_points[background_points==1] = 0
    # print(background_points.shape)
    masks[masks==2] = 0
    seg_weights = class_weights['seg']
    instance_weights = class_weights['instance']
    imgs = imgs.cuda()
    masks = masks.cuda().squeeze(1)
    count = count.cuda()
    # count = count.cuda()
    # output = model.forward(imgs)
    output, count_estimation = model.forward(imgs)
    loss = 0
    
    # count_estimation = count_estimation.view(-1)

    loss_dict = {}
    seg_loss = loss_weights["seg"]*segmentation_loss(output,masks,seg_weights)
    loss_dict["seg_loss"] = seg_loss
    loss +=seg_loss
    # print(f"before: {loss}, seg_loss: {loss_dict['seg_loss']}")
    if "instance" in losses_to_use:
        # inst_loss = loss_weights[0]*instance_loss(output,masks)
        inst_loss = loss_weights["instance"]*instance_loss(output,masks,background_points,instance_weights,imgs)
        loss_dict["inst_loss"] = inst_loss
        loss += inst_loss
    if "convexity" in losses_to_use:
        # cvx_loss = loss_weights[1]*convexity_loss(output,masks)
        cvx_loss = loss_weights["convexity"]*convexity_loss(output,masks)
        loss_dict["cvx_loss"] = cvx_loss
        loss += cvx_loss
    if "circularity" in losses_to_use:
        # circ_loss = loss_weights[2]*circularity_loss(output,masks)
        circ_loss = loss_weights["circularity"]*circularity_loss(output,masks)
        loss_dict["circ_loss"] = circ_loss
        loss += circ_loss
    if "count" in losses_to_use:
        if "count_regress" in losses_to_use:
            closs = loss_weights["count"]*regression_based_count_loss(count_estimation,count)
        elif "count_detect" in losses_to_use:
            closs = loss_weights["count"]*detection_based_count_loss(output,count)
        loss_dict["closs"] = closs
        loss +=closs
    # print(f"after: {loss}, seg_loss: {loss_dict['seg_loss']}")
    return loss, loss_dict
    
