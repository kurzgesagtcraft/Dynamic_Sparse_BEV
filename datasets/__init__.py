# datasets/__init__.py
from .nuscenes_night_dataset import NuScenesNightDataset

def build_dataset(config, split='train'):
    """构建数据集"""
    if config.DATASET.TYPE == 'NuScenesNightDataset':
        # 根据 NuScenesNightDataset 类的实际参数列表进行调用
        return NuScenesNightDataset(
            root_dir=config.DATASET.ROOT,  # 添加 root_dir 参数
            config=config,                 # 传递 config 参数
            split=split                    # 传递 split 参数
            # 删除 night_only 参数，因为类定义中没有这个参数
        )
    else:
        raise ValueError(f"Dataset type {config.DATASET.TYPE} not supported")

    
from torch.utils.data._utils.collate import default_collate

def custom_collate_fn(batch):
    """自动检测并处理所有无法批处理的数据"""
    if not isinstance(batch[0], dict):
        # 如果批次不是字典格式，尝试使用默认collate
        try:
            return default_collate(batch)
        except:
            return batch
    
    data_dict = {}
    
    # 处理字典中的每个键
    for key in batch[0].keys():
        try:
            # 尝试默认批处理
            data_dict[key] = default_collate([item[key] for item in batch])
        except:
            # 如果失败，保持列表形式
            data_dict[key] = [item[key] for item in batch]
            # print(f"键 '{key}' 无法批处理，保持为列表形式")
    
    return data_dict