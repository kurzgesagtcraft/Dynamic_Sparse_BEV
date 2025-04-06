# models/backbones/resnet.py
import torch
import torch.nn as nn
from . import register_backbone

class BasicBlock(nn.Module):
    """ResNet基本块"""
    
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, norm_type='BN'):
        super(BasicBlock, self).__init__()
        
        if norm_type == 'BN':
            norm_layer = nn.BatchNorm2d
        elif norm_type == 'LN':
            norm_layer = lambda x: nn.LayerNorm([x, 1, 1])
        elif norm_type == 'GN':
            norm_layer = lambda x: nn.GroupNorm(32, x)
        else:
            raise ValueError(f"不支持的归一化类型: {norm_type}")
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = norm_layer(out_channels)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class Bottleneck(nn.Module):
    """ResNet瓶颈块"""
    
    expansion = 4
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, norm_type='BN'):
        super(Bottleneck, self).__init__()
        
        if norm_type == 'BN':
            norm_layer = nn.BatchNorm2d
        elif norm_type == 'LN':
            norm_layer = lambda x: nn.LayerNorm([x, 1, 1])
        elif norm_type == 'GN':
            norm_layer = lambda x: nn.GroupNorm(32, x)
        else:
            raise ValueError(f"不支持的归一化类型: {norm_type}")
        
        width = out_channels
        
        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = norm_layer(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = norm_layer(width)
        self.conv3 = nn.Conv2d(width, out_channels * self.expansion, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = norm_layer(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


@register_backbone('ResNet')
class ResNet(nn.Module):
    """ResNet主干网络"""
    
    arch_settings = {
        18: (BasicBlock, [2, 2, 2, 2]),
        34: (BasicBlock, [3, 4, 6, 3]),
        50: (Bottleneck, [3, 4, 6, 3]),
        101: (Bottleneck, [3, 4, 23, 3]),
        152: (Bottleneck, [3, 8, 36, 3])
    }
    
    def __init__(self, config):
        super(ResNet, self).__init__()
        
        # 从配置中获取参数
        depth = config.DEPTH if hasattr(config, 'DEPTH') else 50
        self.in_channels = config.IN_CHANNELS if hasattr(config, 'IN_CHANNELS') else 3
        self.base_channels = config.BASE_CHANNELS if hasattr(config, 'BASE_CHANNELS') else 64
        self.out_indices = config.OUT_INDICES if hasattr(config, 'OUT_INDICES') else (0, 1, 2, 3)
        self.frozen_stages = config.FROZEN_STAGES if hasattr(config, 'FROZEN_STAGES') else -1
        norm_type = config.NORM_TYPE if hasattr(config, 'NORM_TYPE') else 'BN'
        pretrained = config.PATH if hasattr(config, 'PATH') else None
        strides = config.STRIDES if hasattr(config, 'STRIDES') else (1, 2, 2, 2)
        dilations = config.DILATIONS if hasattr(config, 'DILATIONS') else (1, 1, 1, 1)
        
        # 确保depth是整数类型
        if isinstance(depth, (dict, object)) and hasattr(depth, 'value'):
            depth = depth.value
        depth = int(depth)
        
        if depth not in self.arch_settings:
            raise ValueError(f"不支持的ResNet深度: {depth}")
        
        block, layers = self.arch_settings[depth]
        
        if norm_type == 'BN':
            norm_layer = nn.BatchNorm2d
        elif norm_type == 'LN':
            norm_layer = lambda x: nn.LayerNorm([x, 1, 1])
        elif norm_type == 'GN':
            norm_layer = lambda x: nn.GroupNorm(32, x)
        else:
            raise ValueError(f"不支持的归一化类型: {norm_type}")
        
        # 构建输入层
        self.stem = nn.Sequential(
            nn.Conv2d(self.in_channels, self.base_channels, kernel_size=7, stride=2, padding=3, bias=False),
            norm_layer(self.base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # 构建各个阶段
        self.inplanes = self.base_channels
        self.layers = nn.ModuleList()
        
        planes = self.base_channels
        for i in range(len(layers)):
            layer = self._make_layer(
                block=block,
                planes=planes * (2 ** i),
                blocks=layers[i],
                stride=strides[i],
                dilation=dilations[i],
                norm_type=norm_type
            )
            self.layers.append(layer)
        
        # 输出通道数
        self.out_channels = [planes * block.expansion * (2 ** i) for i in range(len(layers))]
        
        # 加载预训练权重
        if pretrained:
            self._load_pretrained_model(pretrained)
        
        # 冻结指定阶段
        self._freeze_stages()
    
    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, norm_type='BN'):
        """构建ResNet层"""
        
        if norm_type == 'BN':
            norm_layer = nn.BatchNorm2d
        elif norm_type == 'LN':
            norm_layer = lambda x: nn.LayerNorm([x, 1, 1])
        elif norm_type == 'GN':
            norm_layer = lambda x: nn.GroupNorm(32, x)
        else:
            raise ValueError(f"不支持的归一化类型: {norm_type}")
        
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion)
            )
        
        layers = []
        layers.append(block(
            in_channels=self.inplanes,
            out_channels=planes,
            stride=stride,
            downsample=downsample,
            norm_type=norm_type
        ))
        
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(
                in_channels=self.inplanes,
                out_channels=planes,
                stride=1,
                norm_type=norm_type
            ))
        
        return nn.Sequential(*layers)
    
    def _freeze_stages(self):
        """冻结指定阶段的参数"""
        if self.frozen_stages >= 0:
            for param in self.stem.parameters():
                param.requires_grad = False
        
        for i in range(min(self.frozen_stages, len(self.layers))):
            layer = self.layers[i]
            layer.eval()
            for param in layer.parameters():
                param.requires_grad = False
    
    def _load_pretrained_model(self, pretrained):
        """加载预训练权重"""
        if isinstance(pretrained, str):
            state_dict = torch.load(pretrained, map_location='cpu')
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            
            # 处理权重名称
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[7:]: v for k, v in state_dict.items()}
            
            self.load_state_dict(state_dict, strict=False)
        else:
            raise TypeError("pretrained 必须是字符串路径")
    
    def forward(self, x):
        """前向传播"""
        x = self.stem(x)
        
        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)
        
        # 如果只需要一个输出，则直接返回
        if len(outs) == 1:
            return outs[0]
        
        return tuple(outs)

@register_backbone('SimpleVoxelNet')
class SimpleVoxelNet(nn.Module):
    """简单的体素化点云处理网络"""
    
    def __init__(self, config):
        super(SimpleVoxelNet, self).__init__()
        
        # 从配置中获取参数
        in_channels = config.IN_CHANNELS if hasattr(config, 'IN_CHANNELS') else 4
        out_channels = config.OUT_CHANNELS if hasattr(config, 'OUT_CHANNELS') else 128
        norm_type = config.NORM_TYPE if hasattr(config, 'NORM_TYPE') else 'BN'
        
        if norm_type == 'BN':
            norm1d = nn.BatchNorm1d
            norm2d = nn.BatchNorm2d
        elif norm_type == 'LN':
            norm1d = nn.LayerNorm
            norm2d = lambda x: nn.LayerNorm([x, 1, 1])
        elif norm_type == 'GN':
            norm1d = lambda x: nn.GroupNorm(min(32, x // 8), x)
            norm2d = lambda x: nn.GroupNorm(min(32, x // 8), x)
        else:
            raise ValueError(f"不支持的归一化类型: {norm_type}")
        
        # 点特征提取
        self.point_encoder = nn.Sequential(
            nn.Linear(in_channels, 32),
            norm1d(32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 64),
            norm1d(64),
            nn.ReLU(inplace=True)
        )
        
        # 点云聚合为BEV
        self.bev_encoder = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            norm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            norm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, out_channels, kernel_size=3, stride=1, padding=1),
            norm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.out_channels = out_channels
    
    def voxelize(self, points, voxel_size, point_cloud_range):
        """将点云体素化"""
        # 点云范围
        x_min, y_min, z_min = point_cloud_range[:3]
        x_max, y_max, z_max = point_cloud_range[3:]
        
        # 体素大小
        vx, vy, vz = voxel_size
        
        # 计算体素网格尺寸
        nx = int((x_max - x_min) / vx)
        ny = int((y_max - y_min) / vy)
        nz = int((z_max - z_min) / vz)
        
        # 筛选范围内的点
        mask = (
            (points[:, 0] >= x_min) & (points[:, 0] < x_max) &
            (points[:, 1] >= y_min) & (points[:, 1] < y_max) &
            (points[:, 2] >= z_min) & (points[:, 2] < z_max)
        )
        
        valid_points = points[mask]
        
        if valid_points.shape[0] == 0:
            return torch.zeros((0, valid_points.shape[1] + 3), device=points.device)
        
        # 计算体素坐标
        voxel_x = ((valid_points[:, 0] - x_min) / vx).long()
        voxel_y = ((valid_points[:, 1] - y_min) / vy).long()
        voxel_z = ((valid_points[:, 2] - z_min) / vz).long()
        
        # 确保坐标在有效范围内
        voxel_x = torch.clamp(voxel_x, 0, nx - 1)
        voxel_y = torch.clamp(voxel_y, 0, ny - 1)
        voxel_z = torch.clamp(voxel_z, 0, nz - 1)
        
        # 生成体素索引
        voxel_idx = voxel_x + voxel_y * nx + voxel_z * nx * ny
        
        # 使用体素索引对点进行排序
        voxel_idx_sort = voxel_idx.argsort()
        sorted_points = valid_points[voxel_idx_sort]
        sorted_voxel_idx = voxel_idx[voxel_idx_sort]
        
        # 找出每个体素的起始点索引
        voxel_starts = (sorted_voxel_idx[1:] != sorted_voxel_idx[:-1]).nonzero(as_tuple=False).reshape(-1)
        voxel_starts = torch.cat([torch.tensor([0], device=points.device), voxel_starts + 1])
        
        # 生成体素特征
        voxel_features = []
        voxel_coords = []
        
        for i in range(len(voxel_starts) - 1):
            start_idx = voxel_starts[i]
            end_idx = voxel_starts[i + 1]
            
            # 获取当前体素中的点
            voxel_points = sorted_points[start_idx:end_idx]
            
            # 计算体素特征（平均值）
            voxel_feature = voxel_points.mean(dim=0)
            
            # 获取体素坐标
            voxel_idx_val = sorted_voxel_idx[start_idx]
            z_coord = voxel_idx_val // (nx * ny)
            y_coord = (voxel_idx_val - z_coord * nx * ny) // nx
            x_coord = voxel_idx_val - z_coord * nx * ny - y_coord * nx
            
            voxel_features.append(voxel_feature)
            voxel_coords.append(torch.tensor([x_coord, y_coord, z_coord], device=points.device))
        
        if len(voxel_features) == 0:
            return torch.zeros((0, valid_points.shape[1] + 3), device=points.device)
        
        voxel_features = torch.stack(voxel_features)
        voxel_coords = torch.stack(voxel_coords)
        
        # 合并特征和坐标
        voxel_data = torch.cat([voxel_coords.float(), voxel_features], dim=1)
        
        return voxel_data
    
    def forward(self, points, voxel_size=[0.2, 0.2, 0.2], point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]):
        """前向传播 - 同时支持张量和列表输入"""
        # 计算BEV尺寸 - 稍后会用到
        x_size = int((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0])
        y_size = int((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1])
        
        # 检查输入类型
        if isinstance(points, list):
            # 处理点云列表（多个样本）
            batch_size = len(points)
            batch_features = []
            
            # 获取设备信息
            device = next(self.parameters()).device
            if batch_size > 0 and len(points[0]) > 0:
                device = points[0].device
            
            # 为每个样本单独处理
            for batch_idx, sample_points in enumerate(points):
                # 跳过空样本
                if sample_points is None or len(sample_points) == 0:
                    batch_features.append(torch.zeros((1, self.out_channels, y_size, x_size), device=device))
                    continue
                
                # 体素化单个样本
                voxel_data = self.voxelize(sample_points, voxel_size, point_cloud_range)
                
                if voxel_data.shape[0] == 0:
                    batch_features.append(torch.zeros((1, self.out_channels, y_size, x_size), device=device))
                    continue
                
                # 提取点特征
                voxel_coords = voxel_data[:, :3]
                voxel_features = voxel_data[:, 3:]
                
                # 点特征编码
                point_features = self.point_encoder(voxel_features)
                
                # 以点特征填充BEV特征图
                bev_features = torch.zeros((1, point_features.shape[1], y_size, x_size), device=sample_points.device)
                
                x_indices = voxel_coords[:, 0].long()
                y_indices = voxel_coords[:, 1].long()
                
                valid_mask = (
                    (x_indices >= 0) & (x_indices < x_size) &
                    (y_indices >= 0) & (y_indices < y_size)
                )
                
                x_indices = x_indices[valid_mask]
                y_indices = y_indices[valid_mask]
                valid_features = point_features[valid_mask]
                
                # 使用scatter_add填充BEV特征图
                indices = y_indices * x_size + x_indices
                for c in range(point_features.shape[1]):
                    channel_features = valid_features[:, c]
                    bev_features[0, c] = bev_features[0, c].reshape(-1).scatter_add(
                        0, indices, channel_features
                    ).reshape(y_size, x_size)
                
                # 编码BEV特征
                sample_output = self.bev_encoder(bev_features)
                batch_features.append(sample_output)
            
            # 合并所有样本的结果
            if not batch_features:
                return torch.zeros((batch_size, self.out_channels, y_size, x_size), device=device)
            
            return torch.cat(batch_features, dim=0)
        
        else:
            # 处理单个张量
            # 将点云体素化
            voxel_data = self.voxelize(points, voxel_size, point_cloud_range)
            
            if voxel_data.shape[0] == 0:
                batch_size = points.shape[0] if len(points.shape) > 2 else 1
                return torch.zeros((batch_size, self.out_channels, y_size, x_size), device=points.device)
            
            # 提取点特征
            voxel_coords = voxel_data[:, :3]
            voxel_features = voxel_data[:, 3:]
            
            # 点特征编码
            point_features = self.point_encoder(voxel_features)
            
            # 以点特征填充BEV特征图
            bev_features = torch.zeros((1, point_features.shape[1], y_size, x_size), device=points.device)
            
            x_indices = voxel_coords[:, 0].long()
            y_indices = voxel_coords[:, 1].long()
            
            valid_mask = (
                (x_indices >= 0) & (x_indices < x_size) &
                (y_indices >= 0) & (y_indices < y_size)
            )
            
            x_indices = x_indices[valid_mask]
            y_indices = y_indices[valid_mask]
            valid_features = point_features[valid_mask]
            
            # 使用scatter_add填充BEV特征图
            indices = y_indices * x_size + x_indices
            for c in range(point_features.shape[1]):
                channel_features = valid_features[:, c]
                bev_features[0, c] = bev_features[0, c].reshape(-1).scatter_add(
                    0, indices, channel_features
                ).reshape(y_size, x_size)
            
            # 编码BEV特征
            bev_output = self.bev_encoder(bev_features)
            
            return bev_output

@register_backbone('SimpleRadarNet')
class SimpleRadarNet(nn.Module):
    """简单的雷达点云处理网络"""
    
    def __init__(self, config):
        super(SimpleRadarNet, self).__init__()
        
        # 从配置中获取参数
        in_channels = config.IN_CHANNELS if hasattr(config, 'IN_CHANNELS') else 5
        out_channels = config.OUT_CHANNELS if hasattr(config, 'OUT_CHANNELS') else 64
        norm_type = config.NORM_TYPE if hasattr(config, 'NORM_TYPE') else 'BN'
        
        if norm_type == 'BN':
            norm2d = nn.BatchNorm2d
        elif norm_type == 'LN':
            norm2d = lambda x: nn.LayerNorm([x, 1, 1])
        elif norm_type == 'GN':
            norm2d = lambda x: nn.GroupNorm(min(32, x // 8), x)
        else:
            raise ValueError(f"不支持的归一化类型: {norm_type}")
        
        # 雷达特征处理
        self.point_encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            norm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            norm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # BEV特征编码
        self.bev_encoder = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            norm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
            norm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.out_channels = out_channels
    
    def forward(self, radar_points_list, voxel_size=[0.4, 0.4, 0.4], point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]):
        """前向传播 - 支持批处理雷达点云列表"""
        batch_size = len(radar_points_list)
        
        # 雷达点云的范围
        x_min, y_min, z_min = point_cloud_range[:3]
        x_max, y_max, z_max = point_cloud_range[3:]
        
        # 构建BEV表示的尺寸
        x_size = int((x_max - x_min) / voxel_size[0])
        y_size = int((y_max - y_min) / voxel_size[1])
        
        # 初始化批次BEV特征图
        device = radar_points_list[0].device if len(radar_points_list) > 0 and len(radar_points_list[0]) > 0 else torch.device('cuda')
        batch_bev_features = torch.zeros((batch_size, self.out_channels, y_size, x_size), device=device)
        
        # 对批次中的每个样本单独处理
        for batch_idx, radar_points in enumerate(radar_points_list):
            if len(radar_points) == 0:
                # 如果没有雷达点，则保持默认的全零特征
                continue
            
            # 筛选范围内的点
            mask = (
                (radar_points[:, 0] >= x_min) & (radar_points[:, 0] < x_max) &
                (radar_points[:, 1] >= y_min) & (radar_points[:, 1] < y_max) &
                (radar_points[:, 2] >= z_min) & (radar_points[:, 2] < z_max)
            )
            
            valid_points = radar_points[mask]
            
            if valid_points.shape[0] == 0:
                # 没有有效点，跳过此样本
                continue
            
            # 计算BEV网格索引
            x_indices = ((valid_points[:, 0] - x_min) / voxel_size[0]).long()
            y_indices = ((valid_points[:, 1] - y_min) / voxel_size[1]).long()
            
            # 确保索引在有效范围内
            x_indices = torch.clamp(x_indices, 0, x_size - 1)
            y_indices = torch.clamp(y_indices, 0, y_size - 1)
            
            # 初始化单个样本的BEV特征图
            sample_bev_features = torch.zeros((1, valid_points.shape[1], y_size, x_size), device=radar_points.device)
            
            # 填充BEV特征图
            for i in range(valid_points.shape[0]):
                x_idx = x_indices[i]
                y_idx = y_indices[i]
                sample_bev_features[0, :, y_idx, x_idx] = valid_points[i]
            
            # 处理雷达特征
            radar_features = self.point_encoder(sample_bev_features)
            
            # 编码BEV特征
            sample_bev_output = self.bev_encoder(radar_features)
            
            # 将结果存储到批次特征图中
            batch_bev_features[batch_idx] = sample_bev_output[0]
        
        return batch_bev_features
