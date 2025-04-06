# models/dynamic_sparse_bev.py
import torch
import torch.nn as nn
from .backbones import build_backbone
from .necks import build_neck
from .heads import build_head
from .sparse_modules import build_sparse_module
from utils.frequency_domain import FrequencyDomainEnhancer

class DynamicSparseBEV(nn.Module):
    """基于频域解耦、可靠性感知与重要性引导的复杂光照环境下BEV感知模型"""
    
    def __init__(self, config):
        super(DynamicSparseBEV, self).__init__()
        self.config = config
        
        # 光照感知层：频域解耦增强网络(FDE-Net)
        self.frequency_enhancer = FrequencyDomainEnhancer(
            in_channels=3,
            low_filter_size=config.FDE.LOW_FILTER_SIZE,
            high_filter_size=config.FDE.HIGH_FILTER_SIZE,
            noise_model_dim=config.FDE.NOISE_MODEL_DIM,
            use_radiation_prior=config.FDE.USE_RADIATION_PRIOR
        )
        
        # 图像与点云骨干网络
        self.img_backbone = build_backbone(config.IMG_BACKBONE)
        self.lidar_backbone = build_backbone(config.LIDAR_BACKBONE) if config.USE_LIDAR else None
        self.radar_backbone = build_backbone(config.RADAR_BACKBONE) if config.USE_RADAR else None
        
        # 动态表征层：可靠性感知稀疏BEV融合(RA-BEV)
        self.view_transformer = build_neck(config.VIEW_TRANSFORMER)
        
        # 动态稀疏化模块
        self.dynamic_sparse = build_sparse_module(
            feature_dim=config.SPARSE.FEATURE_DIM,
            sparsity_threshold=config.SPARSE.THRESHOLD,
            entropy_weight=config.SPARSE.ENTROPY_WEIGHT,
            use_uncertainty=config.SPARSE.USE_UNCERTAINTY
        )
        
        # 异构优化层：重要性引导混合精度量化(IG-MPQ)
        self.use_quantization = config.QUANT.ENABLE
        if self.use_quantization:
            self.register_buffer('light_sensitive_map', torch.ones(1))
            self.light_sensitivity_alpha = config.QUANT.SENSITIVITY_ALPHA
        
        # 检测/分割头
        self.head = build_head(config.HEAD)
        
        # 光照条件评估
        self.light_condition_estimator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(config.IMG_BACKBONE.OUT_CHANNELS, 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, 1),
            nn.Sigmoid()
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def extract_img_features(self, imgs, is_night=None):
        """提取图像特征，应用频域增强"""
        batch_size = imgs[0].shape[0]
        
        # 检测低光照条件
        if is_night is None:
            # 自动评估光照条件
            ref_img = imgs[0]  # 使用前视相机
            light_condition = self.light_condition_estimator(
                self.img_backbone.stem(ref_img)
            )
            is_night = (light_condition < 0.5).float()
        
        # 应用频域解耦增强
        enhanced_imgs = []
        for img in imgs:
            if torch.any(is_night > 0.5):
                # 仅在夜间场景应用增强
                enhanced_img = self.frequency_enhancer(img, light_level=is_night)
                enhanced_imgs.append(enhanced_img)
            else:
                enhanced_imgs.append(img)
        
        # 从增强后的图像提取特征
        img_feats = []
        for img in enhanced_imgs:
            backbone_output = self.img_backbone(img)
            # 如果backbone_output是元组，只取需要的部分
            if isinstance(backbone_output, tuple):
                img_feats.append(backbone_output[0])  # 取第一个元素
            else:
                img_feats.append(backbone_output)
        
        return img_feats, is_night
    
    def extract_lidar_features(self, points):
        """提取激光雷达特征"""
        if self.lidar_backbone is None or points is None:
            return None
        return self.lidar_backbone(points)
    
    def extract_radar_features(self, radar_data):
        """提取毫米波雷达特征"""
        if self.radar_backbone is None or radar_data is None:
            return None
        return self.radar_backbone(radar_data)
    
    def generate_bev_features(self, img_feats, lidar_feats=None, radar_feats=None, light_condition=None):
        """生成BEV特征，使用动态可靠性感知融合"""
        bev_feats = self.view_transformer(
            img_feats=img_feats,
            lidar_feats=lidar_feats,
            radar_feats=radar_feats,
            light_condition=light_condition
        )
        
        # 应用动态稀疏化
        sparse_bev_feats, sparsity_loss, importance_map = self.dynamic_sparse(
            bev_feats, light_condition=light_condition
        )
        
        # 更新光照敏感特征图(用于量化)
        if self.use_quantization and self.training:
            with torch.no_grad():
                current_map = importance_map.detach().mean(0, keepdim=True)
                self.light_sensitive_map = (
                    self.light_sensitive_map * (1 - self.light_sensitivity_alpha) +
                    current_map * self.light_sensitivity_alpha
                )
        
        return sparse_bev_feats, sparsity_loss
    
    def forward(self, batch):
        """前向传播"""
        imgs = batch['img']
        lidar = batch.get('lidar', None)
        radar = batch.get('radar', None)
        is_night = batch.get('is_night', None)
        
        # 1. 光照感知层：特征提取与增强
        img_feats, light_condition = self.extract_img_features(imgs, is_night)
        lidar_feats = self.extract_lidar_features(lidar)
        radar_feats = self.extract_radar_features(radar)
        
        # 2. 动态表征层：BEV生成与稀疏化
        bev_feats, sparsity_loss = self.generate_bev_features(
            img_feats, lidar_feats, radar_feats, light_condition
        )
        
        # 3. 预测层
        predictions = self.head(bev_feats)
        
        if self.training:
            loss_dict = self.head.loss(predictions, batch['gt_labels'])
            loss_dict['sparsity_loss'] = sparsity_loss
            return loss_dict
        else:
            return predictions
    
    def get_light_sensitive_layers(self):
        """获取光照敏感层信息，用于异构精度配置"""
        return self.light_sensitive_map