# models/necks/__init__.py
from typing import Dict, Type

# 存储所有注册的neck类
NECKS: Dict[str, Type] = {}

def register_neck(name: str):
    """
    装饰器，用于注册neck模型
    
    Args:
        name: neck的名称
        
    Returns:
        注册函数的装饰器
    """
    def register_neck_cls(cls):
        if name in NECKS:
            raise ValueError(f"Cannot register duplicate neck ({name})")
        NECKS[name] = cls
        return cls
    return register_neck_cls

def build_neck(config):
    """
    根据配置构建neck
    
    Args:
        config: neck的配置
        
    Returns:
        构造的neck实例
    """
    neck_type = config.NAME
    
    if neck_type not in NECKS:
        raise ValueError(f"Neck {neck_type} not found. Available necks: {list(NECKS.keys())}")
    
    return NECKS[neck_type](config)

# 在这里导入所有neck模型，以确保它们被注册
from .view_transformer import ViewTransformer
# from .fpn import FPN
# from .bifpn import BiFPN
# 等等