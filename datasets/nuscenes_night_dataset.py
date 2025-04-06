# datasets/nuscenes_night_dataset.py
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from pyquaternion import Quaternion
from PIL import Image
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box, LidarPointCloud
from nuscenes.utils.data_classes import RadarPointCloud  # 使用RadarPointCloud而不是LidarPointCloud
from nuscenes.utils.geometry_utils import view_points
from .base_dataset import BaseDataset

class NuScenesNightDataset(BaseDataset):
    """nuScenes夜间数据集
    
    专为复杂光照环境(尤其夜间)设计的数据集加载器
    包含光照条件估计和多模态数据处理
    """
    
    def __init__(self, root_dir, config, split='train', transform=None):
        super(NuScenesNightDataset, self).__init__()
        self.root_dir = root_dir
        self.config = config
        self.split = split
        self.version = config.DATASET.VERSION
        self.transform = transform if transform else self._default_transform()
        
        # 先设置相机信息和传感器使用配置
        self.camera_names = config.DATA.CAMERAS
        self.use_lidar = config.DATA.USE_LIDAR
        self.use_radar = config.DATA.USE_RADAR
        
        # 初始化NuScenes
        self.nusc = NuScenes(
            version=self.version,
            dataroot=self.root_dir,
            verbose=False
        )
        
        # 加载夜间场景信息
        night_info_path = os.path.join(self.root_dir, config.DATASET.NIGHT_INFO_PATH)
        if os.path.exists(night_info_path):
            with open(night_info_path, 'r') as f:
                self.night_info = json.load(f)
        else:
            print(f"警告: 夜间场景信息文件 {night_info_path} 不存在。将视所有场景为日间。")
            self.night_info = {"night_scenes": [], "twilight_scenes": []}
        
        # 筛选样本
        self.samples = self._get_available_samples()
        
        # BEV参数
        self.xbound = config.VIEW_TRANSFORMER.XBOUND
        self.ybound = config.VIEW_TRANSFORMER.YBOUND
        self.zbound = config.VIEW_TRANSFORMER.ZBOUND
        
        # 目标类别
        self.categories = config.DATASET.CLASS_NAMES
        self.category_to_id = {cat: i for i, cat in enumerate(self.categories)}
        
        # 数据增强
        self.augmentation = config.DATA.AUGMENTATION and split == 'train'
        
        print(f"NuScenesNightDataset 初始化完成: {len(self.samples)} 个样本, 分割: {split}")
    
    def _default_transform(self):
        """默认图像变换"""
        if self.split == 'train':
            return transforms.Compose([
                transforms.Resize(self.config.DATA.IMAGE_SIZE),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize(self.config.DATA.IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def _get_available_samples(self):
        """获取可用样本"""
        samples = []
        
        # 根据分割获取场景
        if self.split == 'train':
            scene_splits = ['train']
        elif self.split == 'val':
            scene_splits = ['val']
        elif self.split == 'test':
            scene_splits = ['test']
        else:
            scene_splits = ['train', 'val', 'test']
        
        for scene in self.nusc.scene:
            # 检查场景是否属于当前分割
            scene_split = self.nusc.get('log', scene['log_token'])['logfile'].split('/')[-1].split('_')[0]
            if scene_split not in scene_splits and self.version != 'v1.0-mini':
                continue
            
            # 获取场景的所有样本
            sample_token = scene['first_sample_token']
            while sample_token:
                sample = self.nusc.get('sample', sample_token)
                
                # 检查是否有所有需要的传感器数据
                valid_sample = True
                for cam in self.camera_names:
                    if cam not in sample['data']:
                        valid_sample = False
                        break
                
                if valid_sample:
                    samples.append(sample_token)
                
                sample_token = sample['next']
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def get_image(self, sample_data_token):
        """获取并预处理图像"""
        sample_data = self.nusc.get('sample_data', sample_data_token)
        img_path = os.path.join(self.root_dir, sample_data['filename'])
        img = Image.open(img_path).convert('RGB')
        
        # 应用变换
        if self.transform:
            img = self.transform(img)
        
        return img, sample_data
    
    def get_lidar(self, sample_data_token):
        """获取并处理LiDAR点云"""
        if not self.use_lidar:
            return None
        
        sample_data = self.nusc.get('sample_data', sample_data_token)
        pc_path = os.path.join(self.root_dir, sample_data['filename'])
        
        # 读取点云
        pc = LidarPointCloud.from_file(pc_path)
        points = pc.points.T  # [N, 4]: (x, y, z, intensity)
        
        # 获取校准参数
        cs_record = self.nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
        
        # 应用外参转换到车辆坐标系
        points_xyz = points[:, :3]
        points_features = points[:, 3:]
        
        # 旋转
        rotation = Quaternion(cs_record['rotation']).rotation_matrix
        points_xyz = np.dot(points_xyz, rotation.T)
        
        # 平移
        translation = np.array(cs_record['translation'])
        points_xyz = points_xyz + translation
        
        # 筛选点云范围
        x_min, y_min, z_min = self.config.DATASET.POINT_CLOUD_RANGE[:3]
        x_max, y_max, z_max = self.config.DATASET.POINT_CLOUD_RANGE[3:]
        
        mask = (
            (points_xyz[:, 0] >= x_min) & (points_xyz[:, 0] <= x_max) &
            (points_xyz[:, 1] >= y_min) & (points_xyz[:, 1] <= y_max) &
            (points_xyz[:, 2] >= z_min) & (points_xyz[:, 2] <= z_max)
        )
        
        points_xyz = points_xyz[mask]
        points_features = points_features[mask]
        points = np.concatenate([points_xyz, points_features], axis=1)
        
        return torch.from_numpy(points).float(), sample_data
    
    def get_radar(self, sample_token):
        """获取并处理毫米波雷达数据"""
        if not self.use_radar:
            return None
            
        sample = self.nusc.get('sample', sample_token)
            
        # 合并所有雷达数据
        all_radar_data = []
        radar_tokens = [
            sample['data'].get('RADAR_FRONT', None),
            sample['data'].get('RADAR_FRONT_LEFT', None),
            sample['data'].get('RADAR_FRONT_RIGHT', None),
            sample['data'].get('RADAR_BACK_LEFT', None),
            sample['data'].get('RADAR_BACK_RIGHT', None)
        ]
            
        radar_tokens = [t for t in radar_tokens if t is not None]
            
        for radar_token in radar_tokens:
            radar_data = self.nusc.get('sample_data', radar_token)
            radar_path = os.path.join(self.root_dir, radar_data['filename'])
                
            # 读取雷达点云 - 这里使用RadarPointCloud
            pc = RadarPointCloud.from_file(radar_path)
            points = pc.points.T  # [N, 18]
            
            # 筛选有用的特征: x, y, z, rcs, velocity
            # nuScenes雷达点包含18个维度，我们选择其中5个维度
            points = points[:, [0, 1, 2, 3, 8]]  # x, y, z, rcs, velocity_x
            
            # 获取校准参数
            cs_record = self.nusc.get('calibrated_sensor', radar_data['calibrated_sensor_token'])
            
            # 应用变换
            points_xyz = points[:, :3]
            points_features = points[:, 3:]
            
            # 旋转
            rotation = Quaternion(cs_record['rotation']).rotation_matrix
            points_xyz = np.dot(points_xyz, rotation.T)
            
            # 平移
            translation = np.array(cs_record['translation'])
            points_xyz = points_xyz + translation
            
            # 合并
            points = np.concatenate([points_xyz, points_features], axis=1)
            
            all_radar_data.append(points)
        
        if not all_radar_data:
            # 如果没有雷达数据，返回空张量而不是None
            return torch.zeros((0, 5), dtype=torch.float32)
        
        # 合并所有雷达数据
        all_radar_points = np.concatenate(all_radar_data, axis=0)
        
        # 筛选范围
        x_min, y_min, z_min = self.config.DATASET.POINT_CLOUD_RANGE[:3]
        x_max, y_max, z_max = self.config.DATASET.POINT_CLOUD_RANGE[3:]
        
        mask = (
            (all_radar_points[:, 0] >= x_min) & (all_radar_points[:, 0] <= x_max) &
            (all_radar_points[:, 1] >= y_min) & (all_radar_points[:, 1] <= y_max) &
            (all_radar_points[:, 2] >= z_min) & (all_radar_points[:, 2] <= z_max)
        )
        
        all_radar_points = all_radar_points[mask]
        
        return torch.from_numpy(all_radar_points).float()
    
    def get_calibration(self, sample_data):
        """获取相机标定参数"""
        # 获取标定数据
        cs_record = self.nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
        
        # 内参
        intrinsic = np.array(cs_record['camera_intrinsic'])
        
        # 外参
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = Quaternion(cs_record['rotation']).rotation_matrix
        extrinsic[:3, 3] = np.array(cs_record['translation'])
        
        return torch.from_numpy(intrinsic).float(), torch.from_numpy(extrinsic).float()
    
    def get_annotations(self, sample_token):
        """获取3D目标框标注"""
        sample = self.nusc.get('sample', sample_token)
        
        # 获取所有标注
        annotations = []
        
        for ann_token in sample['anns']:
            ann = self.nusc.get('sample_annotation', ann_token)
            
            # 检查类别是否在目标类别中
            category_name = ann['category_name'].split('.')[0]
            if category_name not in self.categories:
                continue
            
            # 创建3D目标框
            box = Box(
                ann['translation'],
                ann['size'],
                Quaternion(ann['rotation'])
            )
            
            # 目标框属性
            annotation = {
                'sample_token': sample_token,
                'translation': box.center,
                'size': box.wlh,
                'rotation': box.orientation.elements,
                'velocity': ann['velocity'],
                'category_id': self.category_to_id[category_name],
                'category_name': category_name
            }
            
            annotations.append(annotation)
        
        return annotations
    
    def check_is_night(self, sample_token):
        """检查样本是否为夜间场景"""
        sample = self.nusc.get('sample', sample_token)
        scene = self.nusc.get('scene', sample['scene_token'])
        scene_name = scene['name']
        
        is_night = scene_name in self.night_info.get('night_scenes', [])
        is_twilight = scene_name in self.night_info.get('twilight_scenes', [])
        
        if is_night:
            return 1.0  # 夜间
        elif is_twilight:
            return 0.5  # 黄昏
        else:
            return 0.0  # 日间
    
    def __getitem__(self, idx):
        """获取数据项"""
        sample_token = self.samples[idx]
        sample = self.nusc.get('sample', sample_token)
        
        # 检查光照条件
        is_night = self.check_is_night(sample_token)
        
        # 获取相机图像
        images = []
        intrinsics = []
        extrinsics = []
        
        for cam_name in self.camera_names:
            cam_token = sample['data'][cam_name]
            img, sample_data = self.get_image(cam_token)
            images.append(img)
            
            # 获取校准参数
            intrinsic, extrinsic = self.get_calibration(sample_data)
            intrinsics.append(intrinsic)
            extrinsics.append(extrinsic)
        
        # 获取LiDAR点云
        lidar_token = sample['data'].get('LIDAR_TOP', None)
        lidar_data = self.get_lidar(lidar_token) if lidar_token else None
        
        # 获取雷达数据
        radar_data = self.get_radar(sample_token)
        
        # 获取标注
        annotations = self.get_annotations(sample_token)
        
        # 转换标注为目标检测格式
        gt_boxes = []
        gt_labels = []
        
        for ann in annotations:
            # 中心点坐标(x, y, z)
            center = ann['translation']
            
            # 目标尺寸(w, l, h)
            size = ann['size']
            
            # 旋转角(四元数)
            rotation = ann['rotation']
            
            # 速度
            velocity = ann['velocity']
            
            # 类别ID
            class_id = ann['category_id']
            
            # 计算旋转角(弧度)
            yaw = Quaternion(rotation).yaw_pitch_roll[0]
            
            # 合并目标框参数
            box = np.array([
                center[0], center[1], center[2],  # 中心点
                size[0], size[1], size[2],        # 尺寸(w, l, h)
                yaw,                              # 旋转角
                velocity[0], velocity[1]          # 速度(vx, vy)
            ])
            
            gt_boxes.append(box)
            gt_labels.append(class_id)
        
        # 如果没有目标框，添加一个虚拟框
        if len(gt_boxes) == 0:
            gt_boxes = np.zeros((0, 10))
            gt_labels = np.zeros(0)
        else:
            gt_boxes = np.stack(gt_boxes, axis=0)
            gt_labels = np.array(gt_labels)
        
        result = {
            'img': images,
            'lidar': lidar_data[0] if lidar_data else None,
            'radar': radar_data,
            'intrinsics': intrinsics,
            'extrinsics': extrinsics,
            'gt_boxes': torch.from_numpy(gt_boxes).float(),
            'gt_labels': torch.from_numpy(gt_labels).long(),
            'is_night': torch.tensor([is_night]).float(),
            'sample_token': sample_token
        }
        
        return result