# models/heads/__init__.py
from typing import Dict, Type

# 存储所有注册的head类
HEADS: Dict[str, Type] = {}

def register_head(name: str):
    """
    装饰器，用于注册检测头模型
    
    Args:
        name: head的名称
        
    Returns:
        注册函数的装饰器
    """
    def register_head_cls(cls):
        if name in HEADS:
            raise ValueError(f"Cannot register duplicate head ({name})")
        HEADS[name] = cls
        return cls
    return register_head_cls

def build_head(config):
    """
    根据配置构建检测头
    
    Args:
        config: head的配置
        
    Returns:
        构造的head实例
    """
    head_type = config.NAME
    
    if head_type not in HEADS:
        raise ValueError(f"Head {head_type} not found. Available heads: {list(HEADS.keys())}")
    
    return HEADS[head_type](config)

# 在这里导入所有head模型，以确保它们被注册
from .simple_head import SimpleHead
# from .detection_head import DetectionHead
# from .segmentation_head import SegmentationHead
# 等等