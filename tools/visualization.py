# tools/visualization.py
# tool/visualization.py
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap
from nuscenes.utils.data_classes import Box

class SparseBEVVisualizer:
    """动态稀疏BEV可视化工具
    
    用于可视化多模态BEV表示、检测结果、稀疏性和量化效果
    """
    
    def __init__(self, cfg, output_dir=None):
        self.cfg = cfg
        self.output_dir = output_dir or os.path.join(cfg.OUTPUT_DIR, 'visualizations')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 检测类别
        self.categories = cfg.DATASET.CLASS_NAMES
        self.category_colors = self._get_category_colors()
        
        # BEV范围
        self.xbound = cfg.VIEW_TRANSFORMER.XBOUND
        self.ybound = cfg.VIEW_TRANSFORMER.YBOUND
        self.zbound = cfg.VIEW_TRANSFORMER.ZBOUND
        
        # BEV分辨率
        self.bev_resolution = (
            int((self.xbound[1] - self.xbound[0]) / self.xbound[2]),
            int((self.ybound[1] - self.ybound[0]) / self.ybound[2])
        )
        
        # 多模态融合类型
        self.fusion_mode = cfg.VIEW_TRANSFORMER.FUSION_MODE
    
    def _get_category_colors(self):
        """获取类别对应的颜色"""
        colors = plt.cm.rainbow(np.linspace(0, 1, len(self.categories)))
        return {cat: colors[i][:3] for i, cat in enumerate(self.categories)}
    
    def visualize_detection(self, batch, outputs, sample_idx=0, save_path=None):
        """可视化检测结果"""
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        
        # 设置BEV图像范围
        ax.set_xlim(self.xbound[0], self.xbound[1])
        ax.set_ylim(self.ybound[0], self.ybound[1])
        
        # 绘制网格
        ax.grid(which='both', color='lightgray', linestyle='-', alpha=0.5)
        ax.set_axisbelow(True)
        
        # 绘制坐标轴
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('BEV Detection Visualization')
        
        # 绘制真实目标框
        if 'gt_boxes' in batch:
            gt_boxes = batch['gt_boxes'][sample_idx]
            gt_classes = batch['gt_classes'][sample_idx]
            
            for box, cls_id in zip(gt_boxes, gt_classes):
                if cls_id < 0:  # 跳过无效类别
                    continue
                    
                # 提取框参数
                x, y, z, w, l, h, sin_yaw, cos_yaw = box
                yaw = np.arctan2(sin_yaw, cos_yaw)
                
                # 创建矩形
                rect = self._create_box_patch(
                    x, y, w, l, yaw,
                    edgecolor='green',
                    alpha=0.7,
                    linewidth=2,
                    label=f"GT: {self.categories[int(cls_id)]}"
                )
                ax.add_patch(rect)
        
        # 绘制预测目标框
        if 'boxes' in outputs:
            pred_boxes = outputs['boxes'][sample_idx]
            pred_scores = outputs['scores'][sample_idx]
            pred_classes = outputs['classes'][sample_idx]
            
            for box, score, cls_id in zip(pred_boxes, pred_scores, pred_classes):
                if score < 0.3:  # 忽略低置信度的检测
                    continue
                    
                # 提取框参数
                x, y, z, w, l, h, sin_yaw, cos_yaw = box[:8]
                yaw = np.arctan2(sin_yaw, cos_yaw)
                
                # 获取类别颜色
                cat_name = self.categories[int(cls_id)]
                color = self.category_colors.get(cat_name, (1, 0, 0))
                
                # 创建矩形
                rect = self._create_box_patch(
                    x, y, w, l, yaw,
                    edgecolor=color,
                    fill=True,
                    alpha=0.3,
                    linewidth=2,
                    label=f"{cat_name}: {score:.2f}"
                )
                ax.add_patch(rect)
        
        # 显示或保存图像
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close(fig)
    
    def _create_box_patch(self, x, y, w, l, yaw, **kwargs):
        """创建2D边界框补丁"""
        from matplotlib.patches import Rectangle, Polygon
        import matplotlib.transforms as transforms
        
        # 旋转后的矩形顶点
        corners = np.array([
            [-l/2, -w/2],
            [l/2, -w/2],
            [l/2, w/2],
            [-l/2, w/2]
        ])
        
        # 旋转
        rot_mat = np.array([
            [np.cos(yaw), -np.sin(yaw)],
            [np.sin(yaw), np.cos(yaw)]
        ])
        corners = corners @ rot_mat.T
        
        # 平移
        corners[:, 0] += x
        corners[:, 1] += y
        
        return Polygon(corners, **kwargs)
    
    def visualize_bev_feature(self, bev_feature, sample_idx=0, save_path=None):
        """可视化BEV特征图"""
        if isinstance(bev_feature, torch.Tensor):
            bev_feature = bev_feature.detach().cpu().numpy()
        
        # 选择指定样本
        if len(bev_feature.shape) == 4:  # [B, C, H, W]
            feature = bev_feature[sample_idx]
        else:
            feature = bev_feature
        
        # 特征维度降维可视化
        if len(feature.shape) == 3:  # [C, H, W]
            # 使用PCA或直接求和
            feature_map = np.sum(np.abs(feature), axis=0)
        else:
            feature_map = feature
        
        # 归一化
        vmax = np.max(feature_map)
        vmin = np.min(feature_map)
        normalized_map = (feature_map - vmin) / (vmax - vmin + 1e-10)
        
        # 可视化
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # 使用热力图显示
        im = ax.imshow(
            normalized_map,
            cmap='viridis',
            extent=[self.xbound[0], self.xbound[1], self.ybound[0], self.ybound[1]],
            origin='lower'
        )
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Feature Intensity')
        
        # 设置标题和坐标轴
        ax.set_title('BEV Feature Visualization')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        
        # 显示或保存图像
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close(fig)
    
    def visualize_multi_modal_fusion(self, batch, outputs, sample_idx=0, save_path=None):
        """可视化多模态融合效果"""
        # 创建图像网格
        fig = plt.figure(figsize=(18, 12))
        grid_spec = fig.add_gridspec(2, 3)
        
        # 相机图像
        ax_cam = fig.add_subplot(grid_spec[0, 0])
        if 'images' in batch and 'camera_names' in batch:
            # 默认使用前视相机
            cam_idx = batch['camera_names'].index('CAM_FRONT') if 'CAM_FRONT' in batch['camera_names'] else 0
            img = batch['images'][sample_idx][cam_idx].detach().cpu().numpy()
            
            # 转换为RGB并归一化
            img = np.transpose(img, (1, 2, 0))
            img = (img - img.min()) / (img.max() - img.min() + 1e-10)
            
            ax_cam.imshow(img)
            ax_cam.set_title('Camera Image (Front)')
            ax_cam.axis('off')
        
        # LiDAR点云
        ax_lidar = fig.add_subplot(grid_spec[0, 1])
        if 'lidar_points' in batch:
            points = batch['lidar_points'][sample_idx].detach().cpu().numpy()
            
            # 提取x,y坐标和强度
            x = points[:, 0]
            y = points[:, 1]
            intensity = points[:, 3] if points.shape[1] > 3 else np.ones_like(x)
            
            # 绘制点云俯视图
            scatter = ax_lidar.scatter(
                x, y, c=intensity,
                cmap='plasma',
                s=1, alpha=0.5
            )
            
            # 设置范围
            ax_lidar.set_xlim(self.xbound[0], self.xbound[1])
            ax_lidar.set_ylim(self.ybound[0], self.ybound[1])
            
            ax_lidar.set_title('LiDAR Point Cloud')
            ax_lidar.set_xlabel('X (m)')
            ax_lidar.set_ylabel('Y (m)')
            
            # 添加颜色条
            cbar = plt.colorbar(scatter, ax=ax_lidar)
            cbar.set_label('Intensity')
        
        # 雷达点
        ax_radar = fig.add_subplot(grid_spec[0, 2])
        if 'radar_points' in batch:
            points = batch['radar_points'][sample_idx].detach().cpu().numpy()
            
            # 提取x,y坐标和速度
            x = points[:, 0]
            y = points[:, 1]
            velocity = points[:, 4] if points.shape[1] > 4 else np.zeros_like(x)
            
            # 绘制雷达点
            scatter = ax_radar.scatter(
                x, y, c=velocity,
                cmap='coolwarm',
                s=10, alpha=0.8,
                vmin=-10, vmax=10
            )
            
            # 设置范围
            ax_radar.set_xlim(self.xbound[0], self.xbound[1])
            ax_radar.set_ylim(self.ybound[0], self.ybound[1])
            
            ax_radar.set_title('Radar Points (velocity)')
            ax_radar.set_xlabel('X (m)')
            ax_radar.set_ylabel('Y (m)')
            
            # 添加颜色条
            cbar = plt.colorbar(scatter, ax=ax_radar)
            cbar.set_label('Velocity (m/s)')
        
        # 融合BEV特征
        ax_bev = fig.add_subplot(grid_spec[1, :])
        if 'bev_features' in outputs:
            self.visualize_bev_feature(
                outputs['bev_features'],
                sample_idx=sample_idx,
                save_path=None  # 不单独保存
            )
            
            # 再绘制检测结果
            if 'boxes' in outputs:
                pred_boxes = outputs['boxes'][sample_idx]
                pred_scores = outputs['scores'][sample_idx]
                pred_classes = outputs['classes'][sample_idx]
                
                for box, score, cls_id in zip(pred_boxes, pred_scores, pred_classes):
                    if score < 0.3:  # 忽略低置信度的检测
                        continue
                        
                    # 提取框参数
                    x, y, z, w, l, h, sin_yaw, cos_yaw = box[:8]
                    yaw = np.arctan2(sin_yaw, cos_yaw)
                    
                    # 获取类别颜色
                    cat_name = self.categories[int(cls_id)]
                    color = self.category_colors.get(cat_name, (1, 0, 0))
                    
                    # 创建矩形
                    rect = self._create_box_patch(
                        x, y, w, l, yaw,
                        edgecolor=color,
                        fill=True,
                        alpha=0.5,
                        linewidth=2,
                        label=f"{cat_name}: {score:.2f}"
                    )
                    ax_bev.add_patch(rect)
            
            ax_bev.set_title('Fused BEV Representation with Detections')
            ax_bev.set_xlabel('X (m)')
            ax_bev.set_ylabel('Y (m)')
        
        # 调整布局
        plt.tight_layout()
        
        # 显示或保存图像
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close(fig)
    
    def visualize_sparsity(self, sparsity_map, lighting_condition='day', save_path=None):
        """可视化稀疏性分布"""
        if isinstance(sparsity_map, torch.Tensor):
            sparsity_map = sparsity_map.detach().cpu().numpy()
        
        # 确保是2D图像
        if len(sparsity_map.shape) > 2:
            if len(sparsity_map.shape) == 3:  # [H, W, C]
                sparsity_map = sparsity_map.mean(axis=2)
            elif len(sparsity_map.shape) == 4:  # [B, C, H, W]
                sparsity_map = sparsity_map[0].mean(axis=0)
        
        # 计算整体稀疏度
        sparsity_ratio = (sparsity_map < self.cfg.SPARSE.THRESHOLD).sum() / sparsity_map.size
        
        # 可视化
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 创建自定义颜色映射
        colors = [(0.1, 0.1, 0.1), (0.5, 0, 0), (1, 0, 0), (1, 0.5, 0), (1, 1, 0)]
        cmap_name = 'sparse_cmap'
        cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=100)
        
        # 绘制稀疏性热力图
        im = ax.imshow(
            sparsity_map,
            cmap=cm,
            extent=[self.xbound[0], self.xbound[1], self.ybound[0], self.ybound[1]],
            origin='lower',
            vmin=0, vmax=1
        )
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Sparsity Value (0=sparse, 1=dense)')
        
        # 设置标题和坐标轴
        ax.set_title(f'Sparsity Map ({lighting_condition.capitalize()}, Overall: {sparsity_ratio:.2%} sparse)')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        
        # 显示或保存图像
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close(fig)
    
    def visualize_quantization(self, sensitivity_map, bit_allocation, save_path=None):
        """可视化量化敏感度和位宽分布"""
        if not self.cfg.QUANT.ENABLE:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        
        # 敏感度热力图
        if isinstance(sensitivity_map, torch.Tensor):
            sensitivity_map = sensitivity_map.detach().cpu().numpy()
        
        # 确保是2D图像
        if len(sensitivity_map.shape) > 2:
            if len(sensitivity_map.shape) == 3:  # [H, W, C]
                sensitivity_map = sensitivity_map.mean(axis=2)
            elif len(sensitivity_map.shape) == 4:  # [B, C, H, W]
                sensitivity_map = sensitivity_map[0].mean(axis=0)
        
        # 绘制敏感度热力图
        im = ax1.imshow(
            sensitivity_map,
            cmap='hot',
            extent=[self.xbound[0], self.xbound[1], self.ybound[0], self.ybound[1]],
            origin='lower',
            vmin=0, vmax=1
        )
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax1)
        cbar.set_label('Sensitivity to Lighting Conditions')
        
        # 设置标题和坐标轴
        ax1.set_title('Lighting Sensitivity Map')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        
        # 位宽分配饼图
        if bit_allocation:
            labels = [f"{bit}-bit" for bit in bit_allocation.keys()]
            sizes = list(bit_allocation.values())
            
            ax2.pie(
                sizes, labels=labels,
                autopct='%1.1f%%',
                startangle=90,
                colors=plt.cm.tab10.colors[:len(labels)]
            )
            ax2.axis('equal')
            ax2.set_title('Bit-width Allocation')
        
        # 显示或保存图像
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close(fig)
    
    def plot_metrics_by_lighting(self, metrics_history, save_path=None):
        """绘制不同光照条件下的性能指标变化"""
        if not metrics_history:
            return
            
        # 提取指标
        epochs = [entry['epoch'] for entry in metrics_history]
        
        day_map = [entry['day']['mAP'] for entry in metrics_history]
        night_map = [entry['night']['mAP'] for entry in metrics_history]
        twilight_map = [entry['twilight']['mAP'] if 'twilight' in entry else 0 for entry in metrics_history]
        
        day_nds = [entry['day']['NDS'] for entry in metrics_history]
        night_nds = [entry['night']['NDS'] for entry in metrics_history]
        twilight_nds = [entry['twilight']['NDS'] if 'twilight' in entry else 0 for entry in metrics_history]
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        
        # mAP图表
        ax1.plot(epochs, day_map, 'o-', label='Day', color='gold')
        ax1.plot(epochs, night_map, 's-', label='Night', color='navy')
        ax1.plot(epochs, twilight_map, '^-', label='Twilight', color='purple')
        
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('mAP')
        ax1.set_title('mAP by Lighting Condition')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # NDS图表
        ax2.plot(epochs, day_nds, 'o-', label='Day', color='gold')
        ax2.plot(epochs, night_nds, 's-', label='Night', color='navy')
        ax2.plot(epochs, twilight_nds, '^-', label='Twilight', color='purple')
        
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('NDS Score')
        ax2.set_title('NDS by Lighting Condition')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 显示或保存图像
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close(fig)
    
    def plot_sparsity_vs_performance(self, metrics_history, save_path=None):
        """绘制稀疏性与性能的关系"""
        if not metrics_history or 'sparsity' not in metrics_history[0]:
            return
            
        # 提取稀疏度和性能数据
        day_sparsity = [entry['sparsity']['day'] for entry in metrics_history]
        night_sparsity = [entry['sparsity']['night'] for entry in metrics_history]
        
        day_map = [entry['day']['mAP'] for entry in metrics_history]
        night_map = [entry['night']['mAP'] for entry in metrics_history]
        
        # 创建图表
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 绘制散点图
        ax.scatter(day_sparsity, day_map, s=100, label='Day', color='gold', alpha=0.7, edgecolor='black')
        ax.scatter(night_sparsity, night_map, s=100, label='Night', color='navy', alpha=0.7, edgecolor='black')
        
        # 添加趋势线
        if len(day_sparsity) > 1:
            z = np.polyfit(day_sparsity, day_map, 1)
            p = np.poly1d(z)
            ax.plot(day_sparsity, p(day_sparsity), linestyle='--', color='gold')
            
            z = np.polyfit(night_sparsity, night_map, 1)
            p = np.poly1d(z)
            ax.plot(night_sparsity, p(night_sparsity), linestyle='--', color='navy')
        
        ax.set_xlabel('Sparsity Ratio')
        ax.set_ylabel('mAP')
        ax.set_title('Relationship between Sparsity and Performance')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 显示或保存图像
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close(fig)