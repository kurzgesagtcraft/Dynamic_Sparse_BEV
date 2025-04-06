# models/sparse_modules/__init__.py
from typing import Dict, Type

# 存储所有注册的sparse_module类
SPARSE_MODULES: Dict[str, Type] = {}

def register_sparse_module(name: str):
    """
    装饰器，用于注册sparse_module模型
    
    Args:
        name: sparse_module的名称
        
    Returns:
        注册函数的装饰器
    """
    def register_sparse_module_cls(cls):
        if name in SPARSE_MODULES:
            raise ValueError(f"Cannot register duplicate sparse_module ({name})")
        SPARSE_MODULES[name] = cls
        return cls
    return register_sparse_module_cls

def build_sparse_module(feature_dim=None, sparsity_threshold=0.1, entropy_weight=0.01, 
                        use_uncertainty=True, name="DynamicSparseModule", **kwargs):
    """
    根据参数构建sparse_module
    
    Args:
        feature_dim: 特征维度
        sparsity_threshold: 稀疏度阈值
        entropy_weight: 熵权重
        use_uncertainty: 是否使用不确定性
        name: sparse_module的名称，默认为DynamicSparseModule
        **kwargs: 额外的参数
        
    Returns:
        构造的sparse_module实例
    """
    if name not in SPARSE_MODULES:
        raise ValueError(f"Sparse module {name} not found. Available sparse modules: {list(SPARSE_MODULES.keys())}")
    
    # 创建sparse_module实例
    return SPARSE_MODULES[name](
        feature_dim=feature_dim,
        sparsity_threshold=sparsity_threshold,
        entropy_weight=entropy_weight,
        use_uncertainty=use_uncertainty,
        **kwargs
    )

# 在这里导入所有sparse_module模型，以确保它们被注册
from .dynamic_sparse import DynamicSparseModule
# from .static_sparse import StaticSparseModule
# from .pruning_module import PruningModule
# 等等