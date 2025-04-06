# models/backbones/__init__.py
from typing import Dict, Type

# 存储所有注册的backbone类
BACKBONES: Dict[str, Type] = {}

def register_backbone(name: str):
    """
    装饰器，用于注册backbone模型
    
    Args:
        name: backbone的名称
        
    Returns:
        注册函数的装饰器
    """
    def register_backbone_cls(cls):
        if name in BACKBONES:
            raise ValueError(f"Cannot register duplicate backbone ({name})")
        BACKBONES[name] = cls
        return cls
    return register_backbone_cls

def build_backbone(config):
    """
    根据配置构建backbone
    
    Args:
        config: backbone的配置
        
    Returns:
        构造的backbone实例
    """
    # 确保有NAME属性
    if not hasattr(config, 'NAME'):
        raise ValueError("配置中缺少NAME字段，无法确定backbone类型")
    
    backbone_type = config.NAME
    
    # 如果NAME是一个对象，则获取其值
    if hasattr(backbone_type, 'value'):
        backbone_type = backbone_type.value
    
    # 确保backbone_type是字符串类型
    backbone_type = str(backbone_type)
    
    if backbone_type not in BACKBONES:
        raise ValueError(f"Backbone {backbone_type} not found. Available backbones: {list(BACKBONES.keys())}")
    
    return BACKBONES[backbone_type](config)

# 在这里导入所有backbone模型，以确保它们被注册
from .resnet import ResNet, SimpleVoxelNet, SimpleRadarNet