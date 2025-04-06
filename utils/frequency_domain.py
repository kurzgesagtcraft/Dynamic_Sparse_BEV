# utils/frequency_domain.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FrequencyDomainEnhancer(nn.Module):
    """频域解耦增强网络(FDE-Net)
    
    实现了论文中提出的基于频域分解的光照-噪声解耦模型
    数学模型：I^*=F^(-1) [Ψ_l⊙G(FI)+Ψ_h⊙H(F(FI))]
    """
    
    def __init__(self, in_channels=3, low_filter_size=3, high_filter_size=5, 
                 noise_model_dim=16, use_radiation_prior=True):
        super(FrequencyDomainEnhancer, self).__init__()
        self.in_channels = in_channels
        self.use_radiation_prior = use_radiation_prior
        
        # 低频分支 - 可学习低通滤波器
        self.low_pass_conv = nn.Conv2d(
            in_channels * 2,  # 实部和虚部
            in_channels * 2,
            kernel_size=low_filter_size,
            padding=low_filter_size//2,
            groups=in_channels
        )
        
        # 高频分支 - 自适应高频抑制器
        self.high_pass_conv = nn.Conv2d(
            in_channels * 2,
            in_channels * 2,
            kernel_size=high_filter_size,
            padding=high_filter_size//2,
            groups=in_channels
        )
        
        # 频域注意力机制
        self.freq_attention = nn.Sequential(
            nn.Conv2d(in_channels * 4, noise_model_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(noise_model_dim, in_channels * 2, 1),
            nn.Sigmoid()
        )
        
        # 神经形态噪声建模模块
        self.noise_model = nn.Sequential(
            nn.Conv2d(in_channels * 2, noise_model_dim, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(noise_model_dim, noise_model_dim, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(noise_model_dim, in_channels * 2, 3, padding=1)
        )
        
        # 频域条件归一化
        self.fcn_gamma = nn.Parameter(torch.ones(1, in_channels * 2, 1, 1))
        self.fcn_beta = nn.Parameter(torch.zeros(1, in_channels * 2, 1, 1))
        
        # 光照先验
        if use_radiation_prior:
            self.light_embedding = nn.Sequential(
                nn.Linear(1, 16),
                nn.ReLU(inplace=True),
                nn.Linear(16, in_channels * 2)
            )
    
    def fft2d(self, x):
        """2D快速傅里叶变换"""
        # 应用 FFT2
        x_freq = torch.fft.rfft2(x)
        # 将复数张量分解为实部和虚部通道
        x_real = x_freq.real
        x_imag = x_freq.imag
        # 拼接实部和虚部作为通道
        return torch.cat([x_real, x_imag], dim=1)
    
    def ifft2d(self, x_real, x_imag):
        """2D逆快速傅里叶变换"""
        # 创建复数张量
        x_complex = torch.complex(x_real, x_imag)
        # 应用 IFFT2
        return torch.fft.irfft2(x_complex, s=(x_real.shape[2], x_real.shape[3]*2-2))
    
    def forward(self, x, light_level=None):
        """
        Args:
            x: 输入图像 [B, C, H, W]
            light_level: 光照水平估计值 [B, 1] (可选)
        """
        batch_size, _, h, w = x.shape
        
        # 转换到频域
        x_freq = self.fft2d(x)
        
        # 分离实部和虚部
        x_real, x_imag = x_freq.chunk(2, dim=1)
        
        # 低频处理分支 - 主要建模光照补偿
        low_freq = self.low_pass_conv(x_freq)
        
        # 高频处理分支 - 针对噪声抑制
        high_freq = self.high_pass_conv(x_freq)
        
        # 噪声建模
        noise_mask = self.noise_model(high_freq)
        high_freq = high_freq - noise_mask
        
        # 频域注意力，决定低频和高频的权重
        freq_attention = self.freq_attention(torch.cat([low_freq, high_freq], dim=1))
        
        # 应用频域注意力
        balanced_freq = low_freq * freq_attention + high_freq * (1 - freq_attention)
        
        # 应用频域条件归一化
        balanced_freq = balanced_freq * self.fcn_gamma + self.fcn_beta
        
        # 使用光照先验调制频谱
        if self.use_radiation_prior and light_level is not None:
            light_embed = self.light_embedding(light_level.view(batch_size, 1))
            light_embed = light_embed.view(batch_size, -1, 1, 1)
            balanced_freq = balanced_freq * light_embed
        
        # 分离实部和虚部用于逆变换
        balanced_real, balanced_imag = balanced_freq.chunk(2, dim=1)
        
        # 转回空间域
        enhanced = self.ifft2d(balanced_real, balanced_imag)
        
        # 确保输出在合理范围内
        return torch.clamp(enhanced, 0, 1)
