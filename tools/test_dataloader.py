# tools/test_dataloader.py
import os
import sys

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from torch.utils.data import DataLoader
from datasets import build_dataset
from configs.sparse_bev_config import Config

# 设置matplotlib支持中文显示
def setup_chinese_font():
    try:
        # 检查系统中可用的字体
        # 对于Windows系统，优先尝试这些中文字体
        chinese_fonts = ['SimHei', 'Microsoft YaHei', 'STSong', 'SimSun', 'KaiTi']
        
        font_found = False
        for font in chinese_fonts:
            try:
                matplotlib.rc('font', family=font)
                matplotlib.rc('axes', unicode_minus=False)  # 解决负号显示问题
                # 测试该字体是否支持中文
                fig = plt.figure(figsize=(1, 1))
                plt.text(0.5, 0.5, '测试中文', fontsize=12)
                plt.close(fig)
                print(f"使用中文字体: {font}")
                font_found = True
                break
            except:
                continue
        
        if not font_found:
            print("警告: 未找到支持中文的字体，将使用英文标题和标签")
            # 使用英文标题和标签的替代方案
            global use_english_labels
            use_english_labels = True
    except:
        print("警告: 设置中文字体时出错，将使用英文标题和标签")
        use_english_labels = True

# 全局变量，用于控制是否使用英文标签
use_english_labels = False

def test_dataloader():
    """
    测试数据加载器功能
    展示如何加载数据集并创建DataLoader
    """
    # 设置中文字体
    setup_chinese_font()
    
    # 创建配置对象
    config = Config()
    
    # 设置数据集相关配置
    config.dataroot = "data/nuscenes"  # 将由代码自动处理为绝对路径
    config.version = "v1.0-mini"
    config.batch_size = 2
    config.num_workers = 0  # 调试时使用0，生产环境可设为更大值
    config.cameras = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 
                     'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
    config.img_width = 800  # 可根据需要调整
    config.img_height = 450  # 可根据需要调整
    
    # 构建数据集
    print("创建数据集...")
    dataset = build_dataset(config, split='train')
    
    # 检查数据集是否为空
    if len(dataset) == 0:
        print("错误: 数据集为空，无法创建DataLoader")
        return
    
    print(f"数据集大小: {len(dataset)} 个样本")
    
    # 创建数据加载器
    print("创建数据加载器...")
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=dataset.collate_fn
    )
    
    # 测试加载一个批次的数据
    print("加载一个批次数据进行测试...")
    for batch in dataloader:
        # 提取批次数据
        tokens = batch['token']
        images = batch['images']  # [B, N, C, H, W]
        intrinsics = batch['camera_intrinsics']  # [B, N, 3, 3]
        extrinsics = batch['camera_extrinsics']  # [B, N, 4, 4]
        ego_matrices = batch['ego_matrix']  # [B, 4, 4]
        scene_tokens = batch['scene_token']
        annotations = batch['annotations']  # 列表中的列表
        
        # 打印批次信息
        print(f"批次大小: {len(tokens)}")
        print(f"图像形状: {images.shape}")
        print(f"内参矩阵形状: {intrinsics.shape}")
        print(f"外参矩阵形状: {extrinsics.shape}")
        print(f"自车位姿矩阵形状: {ego_matrices.shape}")
        
        # 显示第一个样本的第一个相机图像
        plt.figure(figsize=(10, 6))
        img = images[0, 0].permute(1, 2, 0).cpu().numpy()  # [H, W, C]
        plt.imshow(img)
        
        # 根据是否使用英文标签决定标题
        if use_english_labels:
            plt.title(f"Sample ID: {tokens[0]}, Camera: {config.cameras[0]}")
        else:
            plt.title(f"样本ID: {tokens[0]}, 相机: {config.cameras[0]}")
            
        plt.axis('off')
        
        # 可视化标注（可选）
        if len(annotations[0]) > 0:
            print(f"第一个样本的标注数量: {len(annotations[0])}")
            for i, ann in enumerate(annotations[0][:5]):  # 只显示前5个
                print(f"标注 {i+1}: 类别={ann['category']}, 位置={ann['position']}")
        
        # 如果数据集有BEV遮罩生成，也可以显示
        try:
            bev_mask = dataset.get_bev_mask(tokens[0])
            if bev_mask is not None:
                plt.figure(figsize=(8, 8))
                plt.imshow(bev_mask, cmap='gray')
                
                if use_english_labels:
                    plt.title(f"BEV Mask for Sample ID: {tokens[0]}")
                else:
                    plt.title(f"样本ID: {tokens[0]} 的BEV遮罩")
                    
                plt.axis('off')
        except:
            print("未生成BEV遮罩或生成过程出错")
            
        # 在这里添加额外的可视化或后处理代码...
            
        # 只处理一个批次，然后退出
        plt.show()
        break
        
    print("数据加载器测试完成!")

# 如果直接运行此脚本，则执行测试
if __name__ == "__main__":
    test_dataloader()
