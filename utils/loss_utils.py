import torch
import numpy as np
import torch.nn.functional as F

from utils.box_utils import bbox_iou, xywh2xyxy, xyxy2xywh, generalized_box_iou
from utils.misc import get_world_size


# 构建YOLO格式目标
def build_target(args, gt_bbox, pred, device):
    batch_size = gt_bbox.size(0)
    num_scales = len(pred)
    # 初始化列表存储不同尺度的信息
    coord_list, bbox_list = [], []
    # 对每个尺度进行处理
    for scale_ii in range(num_scales):
        # 计算网格大小
        this_stride = 32 // (2 ** scale_ii)
        grid = args.size // this_stride
        # Convert [x1, y1, x2, y2] to [x_c, y_c, w, h]
        # 转换边界框格式
        center_x = (gt_bbox[:, 0] + gt_bbox[:, 2]) / 2
        center_y = (gt_bbox[:, 1] + gt_bbox[:, 3]) / 2
        box_w = gt_bbox[:, 2] - gt_bbox[:, 0]
        box_h = gt_bbox[:, 3] - gt_bbox[:, 1]
        coord = torch.stack((center_x, center_y, box_w, box_h), dim=1)
        # Normalized by the image size
        # 归一化并缩放到网格尺度
        coord = coord / args.size
        coord = coord * grid
        coord_list.append(coord)
        bbox_list.append(torch.zeros(coord.size(0), 3, 5, grid, grid))

    best_n_list, best_gi, best_gj = [], [], []
    for ii in range(batch_size):
        anch_ious = []
        for scale_ii in range(num_scales):
            this_stride = 32 // (2 ** scale_ii)
            grid = args.size // this_stride
            gw = coord_list[scale_ii][ii, 2]
            gh = coord_list[scale_ii][ii, 3]

            anchor_idxs = [x + 3 * scale_ii for x in [0, 1, 2]]
            anchors = [args.anchors_full[i] for i in anchor_idxs]
            scaled_anchors = [(x[0] / (args.anchor_imsize / grid),
                               x[1] / (args.anchor_imsize / grid)) for x in anchors]

            gt_box = torch.from_numpy(np.array([0, 0, gw.cpu().numpy(), gh.cpu().numpy()])).float().unsqueeze(0)
            ## Get shape of anchor box
            anchor_shapes = torch.FloatTensor(
                np.concatenate((np.zeros((len(scaled_anchors), 2)), np.array(scaled_anchors)), 1))

            ## Calculate iou between gt and anchor shapes
            anch_ious += list(bbox_iou(gt_box, anchor_shapes))
        ## Find the best matching anchor box
        best_n = np.argmax(np.array(anch_ious))
        best_scale = best_n // 3

        best_grid = args.size // (32 / (2 ** best_scale))
        anchor_idxs = [x + 3 * best_scale for x in [0, 1, 2]]
        anchors = [args.anchors_full[i] for i in anchor_idxs]
        scaled_anchors = [(x[0] / (args.anchor_imsize / best_grid), \
                           x[1] / (args.anchor_imsize / best_grid)) for x in anchors]

        gi = coord_list[best_scale][ii, 0].long()
        gj = coord_list[best_scale][ii, 1].long()
        tx = coord_list[best_scale][ii, 0] - gi.float()
        ty = coord_list[best_scale][ii, 1] - gj.float()
        gw = coord_list[best_scale][ii, 2]
        gh = coord_list[best_scale][ii, 3]
        tw = torch.log(gw / scaled_anchors[best_n % 3][0] + 1e-16)
        th = torch.log(gh / scaled_anchors[best_n % 3][1] + 1e-16)

        bbox_list[best_scale][ii, best_n % 3, :, gj, gi] = torch.stack(
            [tx, ty, tw, th, torch.ones(1).to(device).squeeze()])
        best_n_list.append(int(best_n))
        best_gi.append(gi)
        best_gj.append(gj)

    for ii in range(len(bbox_list)):
        bbox_list[ii] = bbox_list[ii].to(device)
    return bbox_list, best_gi, best_gj, best_n_list


def yolo_loss(pred_list, target, gi, gj, best_n_list, device, w_coord=5., w_neg=1. / 5, size_average=True):
    mseloss = torch.nn.MSELoss(size_average=True)
    celoss = torch.nn.CrossEntropyLoss(size_average=True)
    num_scale = len(pred_list)
    batch_size = pred_list[0].size(0)

    # 初始化预测和真实边界框
    pred_bbox = torch.zeros(batch_size, 4).to(device)
    gt_bbox = torch.zeros(batch_size, 4).to(device)
    # 遍历每个样本
    for ii in range(batch_size):
        pred_bbox[ii, 0:2] = torch.sigmoid(
            pred_list[best_n_list[ii] // 3][ii, best_n_list[ii] % 3, 0:2, gj[ii], gi[ii]])
        pred_bbox[ii, 2:4] = pred_list[best_n_list[ii] // 3][ii, best_n_list[ii] % 3, 2:4, gj[ii], gi[ii]]
        gt_bbox[ii, :] = target[best_n_list[ii] // 3][ii, best_n_list[ii] % 3, :4, gj[ii], gi[ii]]
    # 计算坐标损失
    loss_x = mseloss(pred_bbox[:, 0], gt_bbox[:, 0])
    loss_y = mseloss(pred_bbox[:, 1], gt_bbox[:, 1])
    loss_w = mseloss(pred_bbox[:, 2], gt_bbox[:, 2])
    loss_h = mseloss(pred_bbox[:, 3], gt_bbox[:, 3])

    # 计算置信度损失
    pred_conf_list, gt_conf_list = [], []
    for scale_ii in range(num_scale):
        pred_conf_list.append(pred_list[scale_ii][:, :, 4, :, :].contiguous().view(batch_size, -1))
        gt_conf_list.append(target[scale_ii][:, :, 4, :, :].contiguous().view(batch_size, -1))
    pred_conf = torch.cat(pred_conf_list, dim=1)
    gt_conf = torch.cat(gt_conf_list, dim=1)
    loss_conf = celoss(pred_conf, gt_conf.max(1)[1])
    return (loss_x + loss_y + loss_w + loss_h) * w_coord + loss_conf


def trans_vg_loss(batch_pred, batch_target):
    """Compute the losses related to the bounding boxes, 
       including the L1 regression loss and the GIoU loss
    """
    # Transformer视觉定位损失
    batch_size = batch_pred.shape[0]
    # world_size = get_world_size()
    num_boxes = batch_size

    # L1损失：直接回归坐标差异
    loss_bbox = F.l1_loss(batch_pred, batch_target, reduction='none')
    # GIoU损失：计算预测框与真实框的GIoU
    loss_giou = 1 - torch.diag(generalized_box_iou(
        xywh2xyxy(batch_pred),
        xywh2xyxy(batch_target)
    ))
    
    # 计算平均损失
    losses = {}
    losses['loss_bbox'] = loss_bbox.sum() / num_boxes
    losses['loss_giou'] = loss_giou.sum() / num_boxes

    return losses
