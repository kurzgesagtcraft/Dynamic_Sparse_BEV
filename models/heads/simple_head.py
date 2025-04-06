# models/heads/simple_head.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import register_head

@register_head('SimpleHead')
class SimpleHead(nn.Module):
    """简单的BEV检测头
    
    用于3D目标检测，包括分类和边界框回归
    支持多类别和方向分类
    """
    
    def __init__(self, config):
        super(SimpleHead, self).__init__()
        
        # 从配置中获取参数
        self.in_channels = config.IN_CHANNELS
        self.num_classes = config.NUM_CLASSES
        self.class_agnostic = getattr(config, 'CLASS_AGNOSTIC', False)
        self.use_direction_classifier = getattr(config, 'USE_DIRECTION_CLASSIFIER', True)
        
        # 默认回归头
        self.regression_heads = {
            'reg': 2,  # 中心点偏移 (dx, dy)
            'height': 1,  # 高度 (z)
            'dim': 3,  # 尺寸 (w, l, h)
            'rot': 2,  # 旋转 (sin, cos)
            'vel': 2,  # 速度 (vx, vy)
        }
        
        # 使用自定义回归头配置
        if hasattr(config, 'REGRESSION_HEADS'):
            self.regression_heads = config.REGRESSION_HEADS
        
        # 计算回归通道总数
        num_regression_channels = sum(self.regression_heads.values())
        
        # 创建特征提取层
        self.conv_layers = nn.Sequential(
            nn.Conv2d(self.in_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # 创建分类头
        if self.class_agnostic:
            self.cls_head = nn.Conv2d(256, 1, kernel_size=3, padding=1)
        else:
            self.cls_head = nn.Conv2d(256, self.num_classes, kernel_size=3, padding=1)
        
        # 创建回归头
        self.reg_head = nn.Conv2d(256, num_regression_channels, kernel_size=3, padding=1)
        
        # 创建方向分类头
        if self.use_direction_classifier:
            self.dir_head = nn.Conv2d(256, 2, kernel_size=3, padding=1)  # 2个通道：正向/反向
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # 分类头使用更低的初始化值，使训练初期预测更多负样本
        nn.init.constant_(self.cls_head.bias, -4.59)  # -4.59 = log(0.01)
    
    def forward(self, x):
        """前向传播
        
        Args:
            x: BEV特征图，形状为 [B, C, H, W]
        
        Returns:
            字典，包含分类、回归和方向预测结果
        """
        # 特征提取
        feat = self.conv_layers(x)
        
        # 分类预测
        cls_score = self.cls_head(feat)
        
        # 回归预测
        reg_pred = self.reg_head(feat)
        
        # 方向预测
        if self.use_direction_classifier:
            dir_cls = self.dir_head(feat)
        else:
            dir_cls = None
        
        # 拆分回归预测
        ret_dict = {}
        
        # 添加分类预测
        ret_dict['cls_score'] = cls_score
        
        # 添加回归预测
        start_idx = 0
        for k, v in self.regression_heads.items():
            ret_dict[k] = reg_pred[:, start_idx:start_idx+v]
            start_idx += v
        
        # 添加方向预测
        if self.use_direction_classifier:
            ret_dict['dir_cls'] = dir_cls
        
        return ret_dict
    
    def loss(self, predictions, targets):
        """计算损失
        
        Args:
            predictions: 前向传播的预测结果
            targets: 包含真值的字典，包括 'gt_boxes' 和 'gt_labels'
        
        Returns:
            字典，包含各项损失值
        """
        # 获取预测结果
        cls_score = predictions['cls_score']
        batch_size, num_classes, h, w = cls_score.shape
        
        # 获取真值
        gt_boxes = targets['gt_boxes']  # [B, N, 10]
        gt_labels = targets['gt_labels']  # [B, N]
        
        # 准备目标区域
        gt_centers = gt_boxes[..., :2]  # [B, N, 2] - 中心点 (x, y)
        gt_height = gt_boxes[..., 2:3]  # [B, N, 1] - 高度 (z)
        gt_dim = gt_boxes[..., 3:6]  # [B, N, 3] - 尺寸 (w, l, h)
        gt_rot = torch.cat([
            torch.sin(gt_boxes[..., 6:7]),
            torch.cos(gt_boxes[..., 6:7])
        ], dim=-1)  # [B, N, 2] - 旋转 (sin, cos)
        gt_vel = gt_boxes[..., 7:9]  # [B, N, 2] - 速度 (vx, vy)
        
        # 生成目标热图
        heatmap_size = (h, w)
        # 粗略转换真值中心到特征图尺寸
        heatmap = self._generate_heatmap(gt_centers, gt_labels, heatmap_size, batch_size, num_classes)
        
        # 计算分类损失 (focal loss)
        cls_loss = self._focal_loss(cls_score, heatmap)
        
        # 获取正样本索引
        pos_inds = heatmap.eq(1).float()  # [B, C, H, W]
        num_pos = pos_inds.sum().clamp(min=1)
        
        # 计算回归损失
        reg_loss = 0
        
        # 中心点偏移
        if 'reg' in predictions:
            reg_loss += F.l1_loss(predictions['reg'] * pos_inds.unsqueeze(1), 
                               self._get_target_offset(gt_centers, heatmap_size) * pos_inds.unsqueeze(1), 
                               reduction='sum') / num_pos
        
        # 高度
        if 'height' in predictions and 'height' in self.regression_heads:
            reg_loss += F.l1_loss(predictions['height'] * pos_inds.unsqueeze(1),
                               self._gather_target(gt_height, heatmap) * pos_inds.unsqueeze(1),
                               reduction='sum') / num_pos
        
        # 尺寸
        if 'dim' in predictions and 'dim' in self.regression_heads:
            reg_loss += F.l1_loss(predictions['dim'] * pos_inds.unsqueeze(1),
                               self._gather_target(gt_dim, heatmap) * pos_inds.unsqueeze(1),
                               reduction='sum') / num_pos
        
        # 旋转
        if 'rot' in predictions and 'rot' in self.regression_heads:
            reg_loss += F.l1_loss(predictions['rot'] * pos_inds.unsqueeze(1),
                               self._gather_target(gt_rot, heatmap) * pos_inds.unsqueeze(1),
                               reduction='sum') / num_pos
        
        # 速度
        if 'vel' in predictions and 'vel' in self.regression_heads:
            reg_loss += F.l1_loss(predictions['vel'] * pos_inds.unsqueeze(1),
                               self._gather_target(gt_vel, heatmap) * pos_inds.unsqueeze(1),
                               reduction='sum') / num_pos
        
        # 方向分类损失
        dir_loss = 0
        if self.use_direction_classifier and 'dir_cls' in predictions:
            # 获取目标方向类别 (正方向或反方向)
            gt_dir = (gt_boxes[..., 6] > 0).long()  # [B, N]
            dir_target = self._gather_target(gt_dir.unsqueeze(-1), heatmap).squeeze(1)  # [B, H, W]
            dir_loss = F.cross_entropy(predictions['dir_cls'], dir_target, reduction='none')
            dir_loss = (dir_loss * pos_inds.sum(dim=1)).sum() / num_pos
        
        # 合并损失
        loss_dict = {
            'cls_loss': cls_loss,
            'reg_loss': reg_loss
        }
        
        if self.use_direction_classifier:
            loss_dict['dir_loss'] = dir_loss
        
        return loss_dict
    
    def _generate_heatmap(self, gt_centers, gt_labels, heatmap_size, batch_size, num_classes):
        """生成目标热图
        
        Args:
            gt_centers: 目标中心点，形状为 [B, N, 2]
            gt_labels: 目标类别，形状为 [B, N]
            heatmap_size: 热图尺寸 (H, W)
            batch_size: 批次大小
            num_classes: 类别数量
        
        Returns:
            热图，形状为 [B, num_classes, H, W]
        """
        h, w = heatmap_size
        device = gt_centers.device
        
        # 创建热图
        heatmap = torch.zeros((batch_size, num_classes, h, w), device=device)
        
        # 将真值中心缩放到热图尺寸
        # 这里假设真值中心范围为 [-51.2, 51.2]
        range_x = 102.4  # 51.2 * 2
        range_y = 102.4  # 51.2 * 2
        
        for b in range(batch_size):
            for n in range(gt_centers.shape[1]):
                # 跳过填充的目标（标签为0的通常是填充）
                if gt_labels[b, n] < 0:
                    continue
                
                # 获取中心点和类别
                center = gt_centers[b, n]
                cls_id = gt_labels[b, n].long()
                
                # 将中心点坐标映射到热图范围
                center_x = ((center[0] + 51.2) / range_x * w).floor().long()
                center_y = ((center[1] + 51.2) / range_y * h).floor().long()
                
                # 确保中心点在热图范围内
                if center_x < 0 or center_x >= w or center_y < 0 or center_y >= h:
                    continue
                
                # 在中心点位置标记为1
                heatmap[b, cls_id, center_y, center_x] = 1
                
                # 可选：使用高斯核为中心点周围添加平滑权重
                # 这里简化处理，仅将中心点标记为1
        
        return heatmap
    
    def _get_target_offset(self, gt_centers, heatmap_size):
        """计算中心点偏移目标
        
        Args:
            gt_centers: 目标中心点，形状为 [B, N, 2]
            heatmap_size: 热图尺寸 (H, W)
        
        Returns:
            中心点偏移目标，形状为 [B, 2, H, W]
        """
        # 简化实现：直接返回零偏移
        # 实际应用中，应根据真值中心与最近网格中心的偏移计算
        batch_size = gt_centers.shape[0]
        h, w = heatmap_size
        device = gt_centers.device
        
        return torch.zeros((batch_size, 2, h, w), device=device)
    
    def _gather_target(self, gt_values, heatmap):
        """根据热图位置收集目标值
        
        Args:
            gt_values: 目标值，形状为 [B, N, D]
            heatmap: 热图，形状为 [B, C, H, W]
        
        Returns:
            收集的目标值，形状为 [B, D, H, W]
        """
        # 简化实现：直接返回零值
        # 实际应用中，应将gt_values中的值映射到热图对应的位置
        batch_size = gt_values.shape[0]
        dim = gt_values.shape[-1]
        h, w = heatmap.shape[-2:]
        device = gt_values.device
        
        return torch.zeros((batch_size, dim, h, w), device=device)
    
    def _focal_loss(self, pred, target, alpha=2.0, beta=4.0):
        """计算焦点损失
        
        Args:
            pred: 预测结果，形状为 [B, C, H, W]
            target: 目标热图，形状为 [B, C, H, W]
            alpha: 焦点损失的alpha参数
            beta: 焦点损失的beta参数
        
        Returns:
            焦点损失值
        """
        # 应用sigmoid
        pred = torch.sigmoid(pred)
        
        # 计算焦点损失
        pos_inds = target.eq(1).float()
        neg_inds = target.lt(1).float()
        
        neg_weights = torch.pow(1 - target, beta)
        
        loss = 0
        
        # 正样本损失
        pos_loss = torch.log(pred + 1e-8) * torch.pow(1 - pred, alpha) * pos_inds
        
        # 负样本损失
        neg_loss = torch.log(1 - pred + 1e-8) * torch.pow(pred, alpha) * neg_weights * neg_inds
        
        # 计算总损失
        num_pos = pos_inds.sum().clamp(min=1)
        loss = -(pos_loss + neg_loss).sum() / num_pos
        
        return loss
