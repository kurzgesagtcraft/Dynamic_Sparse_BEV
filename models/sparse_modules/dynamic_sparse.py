# models/sparse_modules/dynamic_sparse.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from  . import register_sparse_module

@register_sparse_module('DynamicSparseModule')
class DynamicSparseModule(nn.Module):
    """可靠性感知稀疏BEV融合(RA-BEV)核心模块
    
    实现动态门控机制：G_ijt=σ(⟨Q_it,K_j^t ⟩_d+λ_l⋅I_低光)⋅M_ij^t
    和可微分稀疏度控制(DSC)算法
    """
    
    def __init__(self, feature_dim, sparsity_threshold=0.1, 
                 entropy_weight=0.01, use_uncertainty=True):
        super(DynamicSparseModule, self).__init__()
        self.feature_dim = feature_dim
        self.sparsity_threshold = sparsity_threshold
        self.entropy_weight = entropy_weight
        self.use_uncertainty = use_uncertainty
        
        # 自适应稀疏度控制
        self.sparsity_controller = nn.Sequential(
            nn.Conv2d(feature_dim, 32, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )
        
        # 特征重要性预测
        self.importance_predictor = nn.Sequential(
            nn.Conv2d(feature_dim, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )
        
        # 不确定性建模
        if use_uncertainty:
            self.uncertainty_estimator = nn.Sequential(
                nn.Conv2d(feature_dim, 32, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 1, 1),
                nn.Softplus()
            )
    
    def calculate_entropy(self, x):
        """计算特征的信息熵，用于量化信息含量"""
        # 归一化特征到概率分布
        b, c, h, w = x.shape
        x_flat = x.view(b, c, -1)
        x_norm = F.softmax(x_flat, dim=2)
        
        # 计算每个位置的香农熵
        eps = 1e-8
        entropy = -torch.sum(x_norm * torch.log(x_norm + eps), dim=1)
        return entropy.view(b, 1, h, w)
    
    def forward(self, x, light_condition=None):
        """
        Args:
            x: BEV特征 [B, C, H, W]
            light_condition: 光照条件估计 [B, 1]
        
        Returns:
            sparse_x: 稀疏化后的BEV特征
            sparsity_loss: 稀疏化损失
            importance_map: 特征重要性图
        """
        batch_size, _, h, w = x.shape
        
        # 1. 计算特征信息熵
        feature_entropy = self.calculate_entropy(x)
        
        # 2. 预测特征重要性
        importance_map = self.importance_predictor(x)
        
        # 3. 根据光照条件调整稀疏度
        base_sparsity = self.sparsity_controller(x)
        if light_condition is not None:
            # 低光照条件下降低稀疏度（保留更多信息）
            light_factor = 1.0 - light_condition.view(batch_size, 1, 1, 1)
            adjusted_sparsity = base_sparsity * (1.0 - 0.5 * light_factor)
        else:
            adjusted_sparsity = base_sparsity
        
        # 4. 基于重要性和熵计算稀疏掩码
        combined_importance = importance_map + self.entropy_weight * feature_entropy
        
        # 5. 不确定性建模（可选）
        if self.use_uncertainty:
            uncertainty = self.uncertainty_estimator(x)
            # 不确定性高的区域保留更多特征
            combined_importance = combined_importance * (1.0 + uncertainty)
        
        # 6. 动态调整阈值，保持目标稀疏度
        target_sparsity = adjusted_sparsity.mean()
        sorted_importance, _ = torch.sort(combined_importance.view(batch_size, -1), dim=1, descending=True)
        k = int(h * w * target_sparsity.item())
        threshold = sorted_importance[:, k].view(batch_size, 1, 1, 1)
        
        # 7. 生成二值掩码
        # 使用软阈值进行可微分稀疏化
        temperature = 0.1
        sparse_mask = torch.sigmoid((combined_importance - threshold) / temperature)
        
        # 8. 应用掩码生成稀疏BEV特征
        sparse_x = x * sparse_mask
        
        # 9. 计算稀疏化正则项损失
        mask_ratio = sparse_mask.mean()
        sparsity_loss = F.mse_loss(mask_ratio, target_sparsity)
        
        return sparse_x, sparsity_loss, combined_importance