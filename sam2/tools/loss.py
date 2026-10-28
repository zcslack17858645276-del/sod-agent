import torch
import torch.nn.functional as F
import torch.nn as nn


class BoundaryLoss(nn.Module):
    def __init__(self):
        super(BoundaryLoss, self).__init__()
        # 使用 Sobel 算子
        kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('kernel_x', kernel_x)
        self.register_buffer('kernel_y', kernel_y)

    def forward(self, pred, mask):
        # pred: (B, 1, H, W), 必须先经过 sigmoid
        # mask: (B, 1, H, W), GT
        
        # 提取 GT 的边缘
        edge_mask = F.conv2d(mask, self.kernel_x, padding=1) + \
                    F.conv2d(mask, self.kernel_y, padding=1)
        edge_mask = torch.abs(edge_mask)

        # 提取预测图的边缘
        edge_pred = F.conv2d(pred, self.kernel_x, padding=1) + \
                    F.conv2d(pred, self.kernel_y, padding=1)
        edge_pred = torch.abs(edge_pred)

        # 计算边缘处的 BCE
        # 使用 clamp 防止梯度爆炸
        return F.binary_cross_entropy(edge_pred.clamp(1e-7, 1-1e-7), edge_mask.clamp(1e-7, 1-1e-7))

def optimized_structure_loss(pred, mask):
    # 减小 kernel_size，从 31 降到 7 或 11，对细长物体更友好
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=7, stride=1, padding=3) - mask)
    
    # WBCE
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    # WIoU (修正一下你的公式，使其更标准)
    pred_sig = torch.sigmoid(pred)
    inter = (pred_sig * mask * weit).sum(dim=(2, 3))
    union = (pred_sig * weit).sum(dim=(2, 3)) + (mask * weit).sum(dim=(2, 3)) - inter
    wiou = 1 - (inter + 1e-6) / (union + 1e-6)
    
    return (wbce + wiou).mean()
'''
def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()
'''

def dice_loss(pred, mask):
    pred = torch.sigmoid(pred)
    inter = (pred * mask).sum(dim=(2, 3))
    denom = pred.sum(dim=(2, 3)) + mask.sum(dim=(2, 3))
    dice = (2 * inter + 1) / (denom + 1)
    return (1 - dice).mean()


def iou_loss(pred, mask):
    pred = torch.sigmoid(pred)
    inter = (pred * mask).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + mask.sum(dim=(2, 3)) - inter
    iou = (inter + 1) / (union + 1)
    return (1 - iou).mean()


import torch
import torch.nn.functional as F
import torch.nn as nn


def clip_gradient(optimizer, grad_clip):
    parameters = []
    for group in optimizer.param_groups:
        parameters.extend(group['params'])
    nn.utils.clip_grad_norm_(parameters, grad_clip)


def weighted_bce_loss(pred, mask):
    """
    Weighted BCE Loss (Wei et al. CVPR2020)
    自动增大边缘像素的权重，SOTA 提分核心。
    """
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')

    # 动态计算权重: 局部方差越大(边缘)，权重越大
    ave = F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15)
    weight = 1 + 5 * torch.abs(ave - mask)

    loss = (wbce * weight).sum(dim=(2, 3)) / weight.sum(dim=(2, 3))
    return loss.mean()


def weighted_iou_loss(pred, mask):
    """
    Weighted IoU Loss
    """
    pred = torch.sigmoid(pred)
    inter = (pred * mask).sum(dim=(2, 3))
    union = (pred + mask).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return wiou.mean()


def structure_loss(pred, mask):
    """
    [Updated] Weighted Structure Loss
    """
    if mask.dtype != pred.dtype:
        mask = mask.type_as(pred)

    wbce = weighted_bce_loss(pred, mask)
    wiou = weighted_iou_loss(pred, mask)

    return wbce + wiou