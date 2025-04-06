# utils/lr_scheduler.py
import os
import torch
import torch.distributed as dist
import builtins

def get_rank():
    """
    获取当前进程的rank。
    如果不在分布式环境中，返回0。
    """
    if not dist.is_available() or not dist.is_initialized():
        return 0
    return dist.get_rank()

def get_world_size():
    """
    获取分布式环境中的总进程数。
    如果不在分布式环境中，返回1。
    """
    if not dist.is_available() or not dist.is_initialized():
        return 1
    return dist.get_world_size()

def is_main_process():
    """
    检查当前进程是否为主进程（rank 0）。
    """
    return get_rank() == 0

def init_distributed_mode(args):
    """
    初始化分布式训练环境。
    
    参数:
        args: 包含分布式训练相关参数的对象
    """
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('不使用分布式训练')
        args.distributed = False
        return
    
    args.distributed = True
    
    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print(f'| 分布式初始化 (rank {args.rank}): {args.dist_url}', flush=True)
    
    dist.init_process_group(
        backend=args.dist_backend, 
        init_method=args.dist_url,
        world_size=args.world_size, 
        rank=args.rank
    )
    dist.barrier()
    
    # 确保只有主进程会打印
    if not is_main_process():
        def print_pass(*args, **kwargs):
            pass
        builtins.print = print_pass

def reduce_dict(input_dict, average=True):
    """
    在多个进程之间减少字典中的所有值。
    
    参数:
        input_dict (dict): 所有键值都是张量的字典。
        average (bool): 是否对结果取平均值。
    返回:
        dict: 减少后的字典。
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    
    with torch.no_grad():
        names = []
        values = []
        # 按排序的键迭代，以确保所有进程使用相同的顺序
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        
        if average:
            values /= world_size
            
        reduced_dict = {k: v for k, v in zip(names, values)}
    
    return reduced_dict