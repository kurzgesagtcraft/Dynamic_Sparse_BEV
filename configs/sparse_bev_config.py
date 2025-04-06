# configs/sparse_bev_config.py
import os
from easydict import EasyDict as edict

__C = edict()
cfg = __C

# 基本配置 
__C.TAG = 'dynamic_sparse_bev'
__C.SEED = 42
__C.GPUS = [0]

# 目录配置
__C.ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
__C.CHECKPOINT_DIR = os.path.join(__C.ROOT_DIR, 'checkpoints')
__C.CONFIGS_DIR = os.path.join(__C.ROOT_DIR, 'configs')
__C.DATA_DIR = os.path.join(__C.ROOT_DIR, 'data')
__C.DATASETS_DIR = os.path.join(__C.ROOT_DIR, 'datasets')
__C.MODELS_DIR = os.path.join(__C.ROOT_DIR, 'models')
__C.LOG_DIR = os.path.join(__C.ROOT_DIR, 'log')
__C.TOOLS_DIR = os.path.join(__C.ROOT_DIR, 'tools')
__C.UTILS_DIR = os.path.join(__C.ROOT_DIR, 'utils')

# 数据集配置
__C.DATASET = edict()
__C.DATASET.TYPE = 'NuScenesNightDataset'
__C.DATASET.ROOT = os.path.join(__C.DATA_DIR, 'nuscenes')
__C.DATASET.VERSION = 'v1.0-mini'
__C.DATASET.NIGHT_INFO_PATH = 'night_scene_info.json'
__C.DATASET.NUM_CLASSES = 10
__C.DATASET.CLASS_NAMES = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
    'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
]
__C.DATASET.POINT_CLOUD_RANGE = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
__C.DATASET.VOXEL_SIZE = [0.2, 0.2, 8.0]

# 数据加载配置
__C.DATA = edict()
__C.DATA.BATCH_SIZE = 4
__C.DATA.NUM_WORKERS = 4
__C.DATA.SHUFFLE = True
__C.DATA.IMAGE_SIZE = (900, 1600)  # H x W
__C.DATA.AUGMENTATION = True
__C.DATA.USE_LIDAR = True
__C.DATA.USE_RADAR = True
__C.DATA.CAMERAS = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']

# 训练配置
__C.TRAIN = edict()
__C.TRAIN.EPOCHS = 50
__C.TRAIN.WARMUP_EPOCHS = 5
__C.TRAIN.INITIAL_LR = 0.001
__C.TRAIN.MIN_LR = 1e-6  # 添加最小学习率参数
__C.TRAIN.WARMUP_START_LR = 1e-7  # 添加预热起始学习率
__C.TRAIN.WEIGHT_DECAY = 0.01
__C.TRAIN.MOMENTUM = 0.9
__C.TRAIN.CHECKPOINT_INTERVAL = 5
__C.TRAIN.VAL_INTERVAL = 2
__C.TRAIN.LOG_INTERVAL = 50
__C.TRAIN.AMP = True  # 自动混合精度训练
__C.TRAIN.GRAD_NORM_CLIP = 35.0
__C.TRAIN.OPTIMIZER = 'AdamW'  # 'SGD', 'Adam', 'AdamW'
__C.TRAIN.LR_SCHEDULER = 'cosine'  # 'step', 'multistep', 'cosine', 'warmup_cosine', 'warmup_step', 'plateau'
__C.TRAIN.LR_STEP_SIZE = 30  # 步长学习率调度器的步长
__C.TRAIN.LR_GAMMA = 0.1  # 学习率衰减因子
__C.TRAIN.LR_MILESTONES = [30, 40]  # 多步长学习率调度器的里程碑
__C.TRAIN.PATIENCE = 5  # plateau调度器的耐心参数
__C.TRAIN.RESUME = False
__C.TRAIN.RESUME_PATH = ''
__C.TRAIN.LOSS_WEIGHTS = {
    'cls_loss': 1.0,
    'reg_loss': 2.0,
    'sparsity_loss': 0.1
}
__C.TRAIN.DYNAMIC_SPARSITY = True  # 是否使用动态稀疏化

# 模型配置
__C.MODEL = edict()
__C.MODEL.NAME = 'DynamicSparseBEV'

# 图像骨干网络配置
__C.IMG_BACKBONE = edict()
__C.IMG_BACKBONE.NAME = 'ResNet'
__C.IMG_BACKBONE.PATH = os.path.join(os.path.join(__C.MODELS_DIR, 'backbones'), 'fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth')
__C.IMG_BACKBONE.PRETRAINED = True
__C.IMG_BACKBONE.IN_CHANNELS = 3
__C.IMG_BACKBONE.OUT_CHANNELS = 512
__C.IMG_BACKBONE.NORM_TYPE = 'BN'

# LiDAR骨干网络配置
__C.USE_LIDAR = True
__C.LIDAR_BACKBONE = edict()
__C.LIDAR_BACKBONE.NAME = 'SimpleVoxelNet'
__C.LIDAR_BACKBONE.IN_CHANNELS = 4  # x, y, z, intensity
__C.LIDAR_BACKBONE.OUT_CHANNELS = 128
__C.LIDAR_BACKBONE.NORM_TYPE = 'BN'

# Radar骨干网络配置
__C.USE_RADAR  = True
__C.RADAR_BACKBONE = edict()
__C.RADAR_BACKBONE.NAME = 'SimpleRadarNet'
__C.RADAR_BACKBONE.IN_CHANNELS = 5  # x, y, z, rcs, velocity
__C.RADAR_BACKBONE.OUT_CHANNELS = 64
__C.RADAR_BACKBONE.NORM_TYPE = 'BN'

# 视图变换器配置 
__C.VIEW_TRANSFORMER = edict()
__C.VIEW_TRANSFORMER.NAME = 'ViewTransformer'
__C.VIEW_TRANSFORMER.IMAGE_SIZE = __C.DATA.IMAGE_SIZE
__C.VIEW_TRANSFORMER.FEAT_HEIGHT = 16
__C.VIEW_TRANSFORMER.FEAT_WIDTH = 32
__C.VIEW_TRANSFORMER.XBOUND = [-51.2, 51.2, 0.4]  # min, max, resolution
__C.VIEW_TRANSFORMER.YBOUND = [-51.2, 51.2, 0.4]
__C.VIEW_TRANSFORMER.ZBOUND = [-5.0, 3.0, 0.5]
__C.VIEW_TRANSFORMER.DEPTH_CHANNELS = 32
__C.VIEW_TRANSFORMER.DEPTH_MIN = 1.0
__C.VIEW_TRANSFORMER.DEPTH_MAX = 60.0
__C.VIEW_TRANSFORMER.IMG_FEAT_CHANNELS = __C.IMG_BACKBONE.OUT_CHANNELS
__C.VIEW_TRANSFORMER.LIDAR_FEAT_CHANNELS = __C.LIDAR_BACKBONE.OUT_CHANNELS
__C.VIEW_TRANSFORMER.RADAR_FEAT_CHANNELS = __C.RADAR_BACKBONE.OUT_CHANNELS
__C.VIEW_TRANSFORMER.BEV_FEAT_CHANNELS = 256
__C.VIEW_TRANSFORMER.FUSION_MODE = 'attention'  # 'concat', 'attention'

# 检测头配置
__C.HEAD = edict()
__C.HEAD.NAME = 'SimpleHead'
__C.HEAD.IN_CHANNELS = __C.VIEW_TRANSFORMER.BEV_FEAT_CHANNELS
__C.HEAD.NUM_CLASSES = __C.DATASET.NUM_CLASSES
__C.HEAD.CLASS_AGNOSTIC = False
__C.HEAD.USE_DIRECTION_CLASSIFIER = True
__C.HEAD.REGRESSION_HEADS = {
    'reg': 2,  # offset: (x, y)
    'height': 1,  # height
    'dim': 3,  # (w, l, h)
    'rot': 2,  # (sin, cos)
    'vel': 2,  # (vx, vy)
}

# 频域解耦增强网络(FDE-Net)配置
__C.FDE = edict()
__C.FDE.LOW_FILTER_SIZE = 3
__C.FDE.HIGH_FILTER_SIZE = 5
__C.FDE.NOISE_MODEL_DIM = 16
__C.FDE.USE_RADIATION_PRIOR = True

# 可靠性感知稀疏BEV融合(RA-BEV)配置
__C.SPARSE = edict()
__C.SPARSE.FEATURE_DIM = __C.VIEW_TRANSFORMER.BEV_FEAT_CHANNELS
__C.SPARSE.THRESHOLD = 0.1  # 初始稀疏度
__C.SPARSE.ENTROPY_WEIGHT = 0.01  # 信息熵权重
__C.SPARSE.USE_UNCERTAINTY = True  # 是否使用不确定性建模

# 重要性引导混合精度量化(IG-MPQ)配置
__C.QUANT = edict()
__C.QUANT.ENABLE = True
__C.QUANT.BITWIDTH = [2, 4, 8]  # 可用的位宽列表
__C.QUANT.DEFAULT_BIT = 8  # 默认位宽
__C.QUANT.SENSITIVITY_ALPHA = 0.1  # 光照敏感度学习率
__C.QUANT.SENSITIVITY_THRESHOLD = 0.8  # 高敏感度阈值
__C.QUANT.QAT_EPOCHS = 5  # 量化感知训练轮数

def update_config(cfg, args):
    """根据命令行参数更新配置"""
    if args.config:
        cfg_from_file = __import__(args.config.replace('.py', '').replace('/', '.'))
        cfg.update(cfg_from_file.cfg)
    
    # 更新GPU设置
    if args.gpus:
        cfg.GPUS = [int(i) for i in args.gpus.split(',')]
    
    # 更新其他参数
    if args.batch_size:
        cfg.DATA.BATCH_SIZE = args.batch_size
    
    if args.lr:
        cfg.TRAIN.INITIAL_LR = args.lr
    
    if args.epochs:
        cfg.TRAIN.EPOCHS = args.epochs
    
    if args.tag:
        cfg.TAG = args.tag
    
    # 创建输出目录
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(cfg.CONFIGS_DIR, exist_ok=True)
    os.makedirs(cfg.DATA_DIR, exist_ok=True)
    os.makedirs(cfg.DATASETS_DIR, exist_ok=True)
    os.makedirs(cfg.MODELS_DIR, exist_ok=True)
    os.makedirs(cfg.LOG_DIR, exist_ok=True)
    os.makedirs(cfg.TOOLS_DIR, exist_ok=True)
    os.makedirs(cfg.UTILS_DIR, exist_ok=True)

    return cfg