# tools/test_env.py
import torch
import os
import sys

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_env():
    print("Python版本:", sys.version)
    print("PyTorch版本:", torch.__version__)
    print("CUDA是否可用:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA版本:", torch.version.cuda)
        print("GPU型号:", torch.cuda.get_device_name(0))
        print("GPU数量:", torch.cuda.device_count())
        
        # 测试CUDA内存
        x = torch.rand(10000, 10000).cuda()
        del x
        torch.cuda.empty_cache()
        print("CUDA测试成功!")
    
    # 测试项目导入
    try:
        from models.dynamic_sparse_bev import DynamicSparseBEV
        from configs.sparse_bev_config import Config
        print("项目模块导入测试成功!")
    except ImportError as e:
        print("模块导入失败:", e)

if __name__ == "__main__":
    test_env()
