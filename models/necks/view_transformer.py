# models/necks/view_transformer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import register_neck

@register_neck('ViewTransformer')
class ViewTransformer(nn.Module):
    """可靠性感知BEV视角变换器
    
    实现论文中的BEV表征生成：B_ij=∑_(k=1)^K▒T_k (F_c,F_l)⋅W_ij^k
    包含不确定性感知视角变换机制
    """
    
    def __init__(self, config):
        super(ViewTransformer, self).__init__()
        self.config = config
        self.image_size = config.IMAGE_SIZE
        self.feat_height = config.FEAT_HEIGHT
        self.feat_width = config.FEAT_WIDTH
        self.xbound = config.XBOUND
        self.ybound = config.YBOUND
        self.zbound = config.ZBOUND
        
        self.img_feat_channels = config.IMG_FEAT_CHANNELS
        self.lidar_feat_channels = config.LIDAR_FEAT_CHANNELS if hasattr(config, 'LIDAR_FEAT_CHANNELS') else 0
        self.radar_feat_channels = config.RADAR_FEAT_CHANNELS if hasattr(config, 'RADAR_FEAT_CHANNELS') else 0
        self.bev_feat_channels = config.BEV_FEAT_CHANNELS
        
        # BEV网格参数
        self.dx = (self.xbound[1] - self.xbound[0]) / self.xbound[2]
        self.dy = (self.ybound[1] - self.ybound[0]) / self.ybound[2]
        self.nx = int(self.xbound[2])
        self.ny = int(self.ybound[2])
        
        # 图像到BEV的投影层
        self.img_to_bev = nn.Conv2d(
            self.img_feat_channels,
            self.nx * self.ny,
            kernel_size=1,
            padding=0
        )
        
        # 可学习的深度预测
        self.depth_net = nn.Sequential(
            nn.Conv2d(self.img_feat_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, self.config.DEPTH_CHANNELS, kernel_size=1, padding=0)
        )
        
        # 不确定性预测网络
        self.uncertainty_net = nn.Sequential(
            nn.Conv2d(self.img_feat_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        
        # 多模态融合模块
        self.fusion_mode = getattr(config, 'FUSION_MODE', 'concat')
        fusion_channels = self.img_feat_channels
        if self.lidar_feat_channels > 0:
            fusion_channels += self.lidar_feat_channels
        if self.radar_feat_channels > 0:
            fusion_channels += self.radar_feat_channels
            
        if self.fusion_mode == 'attention':
            self.fusion_attention = nn.Sequential(
                nn.Conv2d(fusion_channels, fusion_channels, kernel_size=1),
                nn.BatchNorm2d(fusion_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(fusion_channels, self.img_feat_channels, kernel_size=1),
                nn.Sigmoid()
            )
        
        # BEV特征整合
        self.bev_encoder = nn.Sequential(
            nn.Conv2d(self.img_feat_channels, self.bev_feat_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.bev_feat_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.bev_feat_channels, self.bev_feat_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.bev_feat_channels),
            nn.ReLU(inplace=True)
        )
        
        # 光照条件适应层
        self.light_adaptor = nn.Sequential(
            nn.Conv2d(self.bev_feat_channels + 1, self.bev_feat_channels, kernel_size=1),
            nn.BatchNorm2d(self.bev_feat_channels),
            nn.ReLU(inplace=True)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def get_geometry(self, intrinsics, extrinsics, offset=0):
        """生成几何变换矩阵"""
        B = intrinsics.shape[0]
        
        # 创建BEV网格坐标
        xcoords = torch.linspace(
            self.xbound[0] + self.dx/2, 
            self.xbound[1] - self.dx/2,
            self.nx, 
            device=intrinsics.device
        )
        ycoords = torch.linspace(
            self.ybound[0] + self.dy/2, 
            self.ybound[1] - self.dy/2,
            self.ny, 
            device=intrinsics.device
        )
        
        # 构建BEV平面上的点
        yy, xx = torch.meshgrid(ycoords, xcoords)
        zz = torch.ones_like(xx) * offset
        
        # 堆叠成坐标点
        bev_points = torch.stack([xx, yy, zz, torch.ones_like(xx)], dim=-1)
        bev_points = bev_points.view(-1, 4).t()
        bev_points = bev_points.unsqueeze(0).repeat(B, 1, 1)  # [B, 4, nx*ny]
        
        # 世界坐标系到相机坐标系的变换
        cam_points = torch.bmm(extrinsics, bev_points)  # [B, 4, nx*ny]
        
        # 相机坐标系到图像平面的投影
        cam_points = cam_points[:, :3, :]  # [B, 3, nx*ny]
        
        # 仅保留相机前方的点
        valid_mask = (cam_points[:, 2, :] > 0).unsqueeze(1)  # [B, 1, nx*ny]
        
        # 投影到图像平面
        img_points = torch.bmm(intrinsics, cam_points)  # [B, 3, nx*ny]
        img_points = img_points[:, :2, :] / (img_points[:, 2:3, :] + 1e-6)  # [B, 2, nx*ny]
        
        # 归一化到[-1, 1]范围用于grid_sample
        img_points = 2 * img_points / torch.tensor(
            [self.image_size[1], self.image_size[0]], 
            device=img_points.device
        ).view(1, 2, 1) - 1
        
        # 转换为sampling grid格式
        img_points = img_points.transpose(1, 2).reshape(B, self.ny, self.nx, 2)  # [B, ny, nx, 2]
        
        # 创建有效掩码
        valid_mask = valid_mask.transpose(1, 2).reshape(B, self.ny, self.nx)  # [B, ny, nx]
        
        return img_points, valid_mask
    
    def transform_cam_to_bev(self, cam_feats, intrinsics, extrinsics, light_condition=None):
        """相机视角转换到BEV视角"""
        B, C, H, W = cam_feats.shape
        
        # 预测深度分布
        depth_prob = F.softmax(self.depth_net(cam_feats), dim=1)  # [B, D, H, W]
        
        # 预测不确定性
        uncertainty = self.uncertainty_net(cam_feats)  # [B, 1, H, W]
        
        # 考虑光照条件
        if light_condition is not None:
            # 低光照条件下增加不确定性
            light_factor = 1.0 - light_condition.view(B, 1, 1, 1)
            uncertainty = uncertainty + 0.2 * light_factor * (1.0 - uncertainty)
        
        # 为每个深度值获取图像平面到BEV的变换
        bev_feats = []
        valid_masks = []
        
        for d_idx in range(self.config.DEPTH_CHANNELS):
            depth = self.config.DEPTH_MIN + d_idx * (
                (self.config.DEPTH_MAX - self.config.DEPTH_MIN) / 
                (self.config.DEPTH_CHANNELS - 1)
            )
            
            # 获取当前深度的几何变换
            img_points, valid_mask = self.get_geometry(intrinsics, extrinsics, offset=depth)
            
            # 采样当前深度的特征
            sampled_feat = F.grid_sample(
                cam_feats, 
                img_points,
                mode='bilinear', 
                padding_mode='zeros', 
                align_corners=True
            )  # [B, C, ny, nx]
            
            # 加权当前深度特征
            weighted_feat = sampled_feat * depth_prob[:, d_idx:d_idx+1]
            
            bev_feats.append(weighted_feat)
            valid_masks.append(valid_mask.unsqueeze(1))
        
        # 沿深度维度聚合
        bev_feat = torch.sum(torch.stack(bev_feats, dim=0), dim=0)  # [B, C, ny, nx]
        valid_mask = torch.cat(valid_masks, dim=1).float().mean(dim=1, keepdim=True)  # [B, 1, ny, nx]
        
        # 应用不确定性
        bev_feat = bev_feat * (1.0 - uncertainty.view(B, 1, 1, 1))
        
        # 应用有效区域掩码
        bev_feat = bev_feat * valid_mask
        
        return bev_feat, uncertainty
    
    def fuse_multimodal_features(self, img_bev, lidar_feats=None, radar_feats=None):
        """融合多模态特征"""
        if lidar_feats is None and radar_feats is None:
            return img_bev
        
        features_to_fuse = [img_bev]
        
        # 添加雷达特征
        if lidar_feats is not None:
            features_to_fuse.append(lidar_feats)
        
        # 添加毫米波雷达特征
        if radar_feats is not None:
            features_to_fuse.append(radar_feats)
        
        # 特征融合
        if self.fusion_mode == 'concat':
            fused_feat = torch.cat(features_to_fuse, dim=1)
            # 使用1x1卷积处理融合特征
            fused_feat = nn.Conv2d(
                fused_feat.shape[1], 
                self.bev_feat_channels, 
                kernel_size=1
            ).to(img_bev.device)(fused_feat)
            
        elif self.fusion_mode == 'attention':
            concat_feat = torch.cat(features_to_fuse, dim=1)
            attention_weights = self.fusion_attention(concat_feat)
            fused_feat = img_bev * attention_weights
        else:
            raise ValueError(f"Unsupported fusion mode: {self.fusion_mode}")
        
        return fused_feat
    
    def forward(self, img_feats, lidar_feats=None, radar_feats=None, light_condition=None):
        """
        Args:
            img_feats: 列表，包含多个相机的图像特征
            lidar_feats: 激光雷达特征 (可选)
            radar_feats: 毫米波雷达特征 (可选)
            light_condition: 光照条件估计值 [B, 1] (可选)
        """
        batch_size = img_feats[0].shape[0]
        
        # 使用第一个相机的内参和外参进行初始化
        device = img_feats[0].device
        intrinsics = torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
        extrinsics = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
        
        # 处理每个相机的特征
        cam_bev_feats = []
        uncertainties = []
        
        for cam_idx, cam_feat in enumerate(img_feats):
            # 为实现简单，这里假设已经设置了内参和外参
            # 实际应用中，应该从数据加载器获取校准参数
            
            # 相机到BEV的转换
            bev_feat, uncertainty = self.transform_cam_to_bev(
                cam_feat, intrinsics, extrinsics, light_condition
            )
            
            cam_bev_feats.append(bev_feat)
            uncertainties.append(uncertainty)
        
        # 聚合多个相机的BEV特征
        if len(cam_bev_feats) > 1:
            # 使用不确定性加权聚合
            uncertainties = torch.stack(uncertainties, dim=0)  # [num_cams, B, 1, H, W]
            confidence = 1.0 - uncertainties
            confidence = confidence / (torch.sum(confidence, dim=0, keepdim=True) + 1e-6)
            
            cam_bev_feats = torch.stack(cam_bev_feats, dim=0)  # [num_cams, B, C, ny, nx]
            weighted_feats = cam_bev_feats * confidence.unsqueeze(2)
            img_bev = torch.sum(weighted_feats, dim=0)  # [B, C, ny, nx]
        else:
            img_bev = cam_bev_feats[0]
        
        # 多模态特征融合
        fused_bev = self.fuse_multimodal_features(img_bev, lidar_feats, radar_feats)
        
        # BEV特征编码
        bev_feat = self.bev_encoder(fused_bev)
        
        # 光照条件适应(如果提供)
        if light_condition is not None:
            # 扩展光照条件到与特征相同的空间尺寸
            light_map = light_condition.view(batch_size, 1, 1, 1).expand(-1, -1, self.ny, self.nx)
            bev_feat = self.light_adaptor(torch.cat([bev_feat, light_map], dim=1))
        
        return bev_feat