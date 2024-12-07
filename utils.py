import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
import numpy as  np
from PIL import Image
from tqdm import tqdm


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Convert logits to probabilities
        probs = F.softmax(inputs, dim=1)

        # Gather the probabilities corresponding to the target class
        targets_one_hot = F.one_hot(targets, num_classes=inputs.size(1))  # Shape: [batch_size, num_classes]
        pt = (probs * targets_one_hot).sum(dim=1)  # Shape: [batch_size]

        # Compute the focal loss
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        loss = -focal_weight * torch.log(pt + 1e-8)

        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def get_warmup_scheduler(optimizer, warm_up_steps, base_lr):
    # Set initial learning rate to base_lr / 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = base_lr / 10

    def warmup_lr_scheduler(step):
        if step < warm_up_steps:
            return step / warm_up_steps  # Gradually scale up
        return 1.0  # Return the base_lr afterward

    return LambdaLR(optimizer, lr_lambda=warmup_lr_scheduler)


def calculate_mean_std(image_paths, batch_size=100, target_size=(224, 224)):
    """
    计算多个图片的整体均值 (mean) 和标准差 (std)，使用增量计算避免内存溢出。

    参数:
    - image_paths: list，包含所有图片路径的列表。
    - batch_size: 每次处理的图片批次大小。
    - target_size: 所有图像调整到的目标尺寸。

    返回:
    - overall_mean: 每个通道的整体均值。
    - overall_std: 每个通道的整体标准差。
    """
    # 初始化统计量
    mean = np.zeros(3)  # 每个通道的均值
    M2 = np.zeros(3)  # 用于计算标准差的平方差
    count = 0

    print(len(image_paths))

    # tqdm 用于显示进度条
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing images"):
        batch_paths = image_paths[i:i + batch_size]  # 处理一批图片
        batch_pixel_values = []

        # 处理当前批次中的图片
        for image_path in batch_paths:
            try:
                image = Image.open(image_path).convert("RGB")
                image = image.resize(target_size)  # 强制调整图片大小
                image_array = np.array(image) / 255.0  # 归一化到 [0, 1]
                batch_pixel_values.append(image_array)
            except Exception as e:
                print(f"无法处理图片: {image_path}, 错误信息: {e}")

        # 确保 batch_pixel_values 中的所有元素形状一致
        batch_pixel_values = np.array(batch_pixel_values)

        # 增量更新均值和标准差
        batch_pixels = batch_pixel_values.reshape(-1, 3)
        for pixel in batch_pixels:
            count += 1
            delta = pixel - mean
            mean += delta / count
            M2 += delta * (pixel - mean)

    # 计算最终的标准差
    variance = M2 / count
    std = np.sqrt(variance)

    return mean, std