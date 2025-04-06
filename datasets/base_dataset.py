# datasets/base_datase
import torch
from torch.utils.data import Dataset
import numpy as np
from abc import ABC, abstractmethod

class BaseDataset(Dataset, ABC):
    """所有数据集的基类
    
    提供常用的数据集接口和实用功能
    """
    
    def __init__(self):
        super(BaseDataset, self).__init__()
    
    @abstractmethod
    def __len__(self):
        """返回数据集中的样本数量"""
        pass
    
    @abstractmethod
    def __getitem__(self, idx):
        """获取数据集中的一个样本"""
        pass
    
    def normalize_points(self, points, pc_range):
        """将点云归一化到指定范围内"""
        # 获取点云范围
        x_min, y_min, z_min = pc_range[:3]
        x_max, y_max, z_max = pc_range[3:]
        
        # 归一化点云
        points_norm = points.clone()
        points_norm[:, 0] = 2 * (points[:, 0] - x_min) / (x_max - x_min) - 1
        points_norm[:, 1] = 2 * (points[:, 1] - y_min) / (y_max - y_min) - 1
        points_norm[:, 2] = 2 * (points[:, 2] - z_min) / (z_max - z_min) - 1
        
        return points_norm
    
    def normalize_box(self, boxes, pc_range):
        """将3D边界框归一化"""
        # 获取点云范围
        x_min, y_min, z_min = pc_range[:3]
        x_max, y_max, z_max = pc_range[3:]
        
        # 归一化边界框中心
        boxes_norm = boxes.clone()
        boxes_norm[:, 0] = 2 * (boxes[:, 0] - x_min) / (x_max - x_min) - 1  # 中心x
        boxes_norm[:, 1] = 2 * (boxes[:, 1] - y_min) / (y_max - y_min) - 1  # 中心y
        boxes_norm[:, 2] = 2 * (boxes[:, 2] - z_min) / (z_max - z_min) - 1  # 中心z
        
        # 归一化边界框尺寸
        boxes_norm[:, 3] = boxes[:, 3] / (x_max - x_min)  # 宽度
        boxes_norm[:, 4] = boxes[:, 4] / (y_max - y_min)  # 长度 
        boxes_norm[:, 5] = boxes[:, 5] / (z_max - z_min)  # 高度
        
        return boxes_norm
    
    def denormalize_box(self, boxes_norm, pc_range):
        """将归一化的3D边界框转换回原始坐标系"""
        # 获取点云范围
        x_min, y_min, z_min = pc_range[:3]
        x_max, y_max, z_max = pc_range[3:]
        
        # 反归一化边界框
        boxes = boxes_norm.clone()
        boxes[:, 0] = (boxes_norm[:, 0] + 1) * (x_max - x_min) / 2 + x_min  # 中心x
        boxes[:, 1] = (boxes_norm[:, 1] + 1) * (y_max - y_min) / 2 + y_min  # 中心y
        boxes[:, 2] = (boxes_norm[:, 2] + 1) * (z_max - z_min) / 2 + z_min  # 中心z
        
        boxes[:, 3] = boxes_norm[:, 3] * (x_max - x_min)  # 宽度
        boxes[:, 4] = boxes_norm[:, 4] * (y_max - y_min)  # 长度
        boxes[:, 5] = boxes_norm[:, 5] * (z_max - z_min)  # 高度
        
        return boxes
    
    def get_bev_coordinates(self, x_bound, y_bound, z_bound=None):
        """获取BEV视图的坐标网格"""
        x_min, x_max, x_res = x_bound
        y_min, y_max, y_res = y_bound
        
        x_size = int((x_max - x_min) / x_res)
        y_size = int((y_max - y_min) / y_res)
        
        x_coords = torch.linspace(x_min + x_res/2, x_max - x_res/2, x_size)
        y_coords = torch.linspace(y_min + y_res/2, y_max - y_res/2, y_size)
        
        # 创建网格坐标
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        return x_grid, y_grid
    
    def transform_points_to_bev(self, points, x_bound, y_bound, z_bound=None):
        """将点云转换为BEV视图表示"""
        x_min, x_max, x_res = x_bound
        y_min, y_max, y_res = y_bound
        
        x_size = int((x_max - x_min) / x_res)
        y_size = int((y_max - y_min) / y_res)
        
        # 过滤范围外的点
        mask = (
            (points[:, 0] >= x_min) & (points[:, 0] < x_max) &
            (points[:, 1] >= y_min) & (points[:, 1] < y_max)
        )
        
        if z_bound is not None:
            z_min, z_max, z_res = z_bound
            mask = mask & (points[:, 2] >= z_min) & (points[:, 2] < z_max)
        
        points = points[mask]
        
        # 计算BEV网格索引
        x_indices = ((points[:, 0] - x_min) / x_res).long()
        y_indices = ((points[:, 1] - y_min) / y_res).long()
        
        # 确保索引在有效范围内
        x_indices = torch.clamp(x_indices, 0, x_size - 1)
        y_indices = torch.clamp(y_indices, 0, y_size - 1)
        
        # 创建BEV特征图
        bev_feature = torch.zeros((y_size, x_size), dtype=torch.float32)
        
        # 填充BEV特征图
        for i in range(points.shape[0]):
            x_idx = x_indices[i]
            y_idx = y_indices[i]
            bev_feature[y_idx, x_idx] += 1
        
        return bev_feature
    
    def augment_data(self, data_dict, augmentations):
        """对数据进行增强
        
        Args:
            data_dict: 包含各种数据的字典
            augmentations: 要应用的增强列表
        
        Returns:
            增强后的数据字典
        """
        # 如果没有指定增强，直接返回原始数据
        if not augmentations:
            return data_dict
        
        # 应用每一种增强
        for aug_func in augmentations:
            data_dict = aug_func(data_dict)
        
        return data_dict
    
    def collate_fn(self, batch):
        """自定义的collate函数，处理不同长度的数据"""
        batch_dict = {}
        
        for key in batch[0].keys():
            if isinstance(batch[0][key], torch.Tensor):
                # 对于张量，尝试堆叠
                try:
                    batch_dict[key] = torch.stack([b[key] for b in batch])
                except:
                    # 如果不能堆叠（不同形状），则存为列表
                    batch_dict[key] = [b[key] for b in batch]
            elif isinstance(batch[0][key], np.ndarray):
                # 对于NumPy数组，先转为张量再尝试堆叠
                try:
                    batch_dict[key] = torch.stack([torch.from_numpy(b[key]) for b in batch])
                except:
                    batch_dict[key] = [torch.from_numpy(b[key]) for b in batch]
            elif isinstance(batch[0][key], list) and len(batch[0][key]) > 0 and isinstance(batch[0][key][0], torch.Tensor):
                # 对于张量列表，分别堆叠每个位置的张量
                batch_dict[key] = []
                for i in range(len(batch[0][key])):
                    try:
                        batch_dict[key].append(torch.stack([b[key][i] for b in batch]))
                    except:
                        batch_dict[key].append([b[key][i] for b in batch])
            else:
                # 对于其他类型，保持为列表
                batch_dict[key] = [b[key] for b in batch]
        
        return batch_dict