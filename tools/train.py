# tools/train.py
import os
import time
import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.sparse_bev_config import cfg, update_config
from datasets import build_dataset
from datasets import custom_collate_fn
from models.dynamic_sparse_bev import DynamicSparseBEV
from utils.logger import setup_logger
from utils.lr_scheduler import build_scheduler
from utils.distributed import get_rank, init_distributed_mode, is_main_process


def parse_args():
    parser = argparse.ArgumentParser(description='训练动态稀疏BEV模型')
    parser.add_argument('--config', type=str, default='', help='配置文件路径')
    parser.add_argument('--gpus', type=str, default='0', help='要使用的GPU，如 "0,1,2"')
    parser.add_argument('--batch-size', type=int, help='批量大小')
    parser.add_argument('--lr', type=float, help='学习率')
    parser.add_argument('--epochs', type=int, help='训练轮数')
    parser.add_argument('--tag', type=str, help='实验标签')
    parser.add_argument('--dist-url', default='env://', help='分布式训练url')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    return args


def build_optimizer(model, config):
    """构建优化器"""
    if config.TRAIN.OPTIMIZER == 'Adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.TRAIN.INITIAL_LR,
            weight_decay=config.TRAIN.WEIGHT_DECAY
        )
    elif config.TRAIN.OPTIMIZER == 'AdamW':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.TRAIN.INITIAL_LR,
            weight_decay=config.TRAIN.WEIGHT_DECAY
        )
    elif config.TRAIN.OPTIMIZER == 'SGD':
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.TRAIN.INITIAL_LR,
            momentum=config.TRAIN.MOMENTUM,
            weight_decay=config.TRAIN.WEIGHT_DECAY
        )
    else:
        raise ValueError(f"不支持的优化器: {config.TRAIN.OPTIMIZER}")
    
    return optimizer


def train_one_epoch(model, dataloader, optimizer, device, epoch, config, scaler=None, logger=None):
    """训练一个epoch"""
    model.train()
    
    num_batches = len(dataloader)
    epoch_loss = 0
    epoch_cls_loss = 0
    epoch_reg_loss = 0
    epoch_sparsity_loss = 0
    
    for batch_idx, batch in enumerate(dataloader):
        # 将数据移动到设备
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
            elif isinstance(v, list):
                batch[k] = [item.to(device) if isinstance(item, torch.Tensor) else item for item in v]
        
        # 前向和反向传播
        optimizer.zero_grad()
        
        if config.TRAIN.AMP:
            with autocast():
                loss_dict = model(batch)
                losses = sum(
                    [v * config.TRAIN.LOSS_WEIGHTS.get(k, 1.0) for k, v in loss_dict.items()]
                )
            
            scaler.scale(losses).backward()
            
            # 梯度裁剪
            if config.TRAIN.GRAD_NORM_CLIP > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.TRAIN.GRAD_NORM_CLIP
                )
            
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_dict = model(batch)
            losses = sum(
                [v * config.TRAIN.LOSS_WEIGHTS.get(k, 1.0) for k, v in loss_dict.items()]
            )
            
            losses.backward()
            
            # 梯度裁剪
            if config.TRAIN.GRAD_NORM_CLIP > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.TRAIN.GRAD_NORM_CLIP
                )
            
            optimizer.step()
        
        # 更新损失
        epoch_loss += losses.item()
        epoch_cls_loss += loss_dict.get('cls_loss', 0).item()
        epoch_reg_loss += loss_dict.get('reg_loss', 0).item()
        epoch_sparsity_loss += loss_dict.get('sparsity_loss', 0).item()
        
        # 日志记录
        if batch_idx % config.TRAIN.LOG_INTERVAL == 0 and logger:
            logger.info(
                f"Epoch [{epoch}/{config.TRAIN.EPOCHS}] "
                f"Step [{batch_idx}/{num_batches}] "
                f"Loss: {losses.item():.4f} "
                f"Cls: {loss_dict.get('cls_loss', 0).item():.4f} "
                f"Reg: {loss_dict.get('reg_loss', 0).item():.4f} "
                f"Sparsity: {loss_dict.get('sparsity_loss', 0).item():.4f}"
            )
    
    # 计算平均损失
    epoch_loss /= num_batches
    epoch_cls_loss /= num_batches
    epoch_reg_loss /= num_batches
    epoch_sparsity_loss /= num_batches
    
    return {
        'loss': epoch_loss,
        'cls_loss': epoch_cls_loss,
        'reg_loss': epoch_reg_loss,
        'sparsity_loss': epoch_sparsity_loss
    }


def validate(model, dataloader, device, config, logger=None):
    """验证模型"""
    model.eval()
    
    num_batches = len(dataloader)
    total_loss = 0
    cls_loss = 0
    reg_loss = 0
    sparsity_loss = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # 将数据移动到设备
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
                elif isinstance(v, list):
                    batch[k] = [item.to(device) if isinstance(item, torch.Tensor) else item for item in v]
            
            # 前向传播
            loss_dict = model(batch)
            losses = sum(
                [v * config.TRAIN.LOSS_WEIGHTS.get(k, 1.0) for k, v in loss_dict.items()]
            )
            
            # 更新损失
            total_loss += losses.item()
            cls_loss += loss_dict.get('cls_loss', 0).item()
            reg_loss += loss_dict.get('reg_loss', 0).item()
            sparsity_loss += loss_dict.get('sparsity_loss', 0).item()
            
            # 日志记录
            if batch_idx % config.TRAIN.LOG_INTERVAL == 0 and logger:
                logger.info(
                    f"Validation Step [{batch_idx}/{num_batches}] "
                    f"Loss: {losses.item():.4f}"
                )
    
    # 计算平均损失
    total_loss /= num_batches
    cls_loss /= num_batches
    reg_loss /= num_batches
    sparsity_loss /= num_batches
    
    return {
        'loss': total_loss,
        'cls_loss': cls_loss,
        'reg_loss': reg_loss,
        'sparsity_loss': sparsity_loss
    }


def save_checkpoint(model, optimizer, lr_scheduler, epoch, config, is_best=False):
    """保存检查点"""
    state = {
        'epoch': epoch,
        'model': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict() if lr_scheduler else None,
        'config': config
    }
    
    filename = os.path.join(config.CHECKPOINT_DIR, f"{config.TAG}_epoch_{epoch}.pth")
    torch.save(state, filename)
    
    if is_best:
        best_filename = os.path.join(config.CHECKPOINT_DIR, f"{config.TAG}_best.pth")
        torch.save(state, best_filename)


def main():
    args = parse_args()
    config = update_config(cfg, args)
    
    # 分布式初始化
    init_distributed_mode(args)
    
    # 设置随机种子
    seed = config.SEED + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = True
    
    # 设置设备
    device = torch.device(f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu")
    
    # 创建日志
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_file = os.path.join(config.LOG_DIR, f"{config.TAG}_{timestamp}.log")
    logger = setup_logger(name=config.TAG, log_file=log_file, distributed_rank=get_rank())
    
    #logger.info(f"配置:\n{config}")
    
    # 创建TensorBoard记录器
    writer = None
    if is_main_process():
        writer = SummaryWriter(log_dir=os.path.join(config.LOG_DIR, f"{config.TAG}_{timestamp}"))
    
    # 创建数据集和数据加载器
    train_dataset = build_dataset(config, split='train')
    val_dataset = build_dataset(config, split='val')
    
    # 创建分布式采样器
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=(train_sampler is None) and config.DATA.SHUFFLE,
        collate_fn=custom_collate_fn,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=True,
        sampler=val_sampler
    )
    
    # 创建模型
    model = DynamicSparseBEV(config)
    model = model.to(device)
    
    # 分布式包装
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], 
            output_device=args.local_rank,
            find_unused_parameters=True
        )
    
    # 创建优化器
    optimizer = build_optimizer(model, config)
    
    # 创建学习率调度器
    lr_scheduler = build_scheduler(config, optimizer, len(train_loader))
    
    # 创建混合精度训练的缩放器
    scaler = GradScaler() if config.TRAIN.AMP else None
    
    # 从检查点恢复
    start_epoch = 0
    if config.TRAIN.RESUME and config.TRAIN.RESUME_PATH:
        if os.path.isfile(config.TRAIN.RESUME_PATH):
            logger.info(f"从检查点恢复: {config.TRAIN.RESUME_PATH}")
            checkpoint = torch.load(config.TRAIN.RESUME_PATH, map_location='cpu')
            
            # 加载模型参数
            if hasattr(model, 'module'):
                model.module.load_state_dict(checkpoint['model'])
            else:
                model.load_state_dict(checkpoint['model'])
            
            # 加载优化器参数
            optimizer.load_state_dict(checkpoint['optimizer'])
            
            # 加载学习率调度器
            if checkpoint['lr_scheduler'] and lr_scheduler:
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            
            # 恢复轮次
            start_epoch = checkpoint['epoch'] + 1
            
            logger.info(f"从epoch {start_epoch}恢复训练")
        else:
            logger.warning(f"未找到检查点: {config.TRAIN.RESUME_PATH}")
    
    # 开始训练
    logger.info("开始训练...")
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, config.TRAIN.EPOCHS):
        # 重置采样器
        if train_sampler:
            train_sampler.set_epoch(epoch)
        
        # 训练一个epoch
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device, epoch, config, scaler, logger
        )
        
        # 更新学习率
        if lr_scheduler:
            lr_scheduler.step()
        
        # 记录训练指标
        if writer and is_main_process():
            for k, v in train_metrics.items():
                writer.add_scalar(f'train/{k}', v, epoch)
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        
        # 保存检查点
        if (epoch + 1) % config.TRAIN.CHECKPOINT_INTERVAL == 0 and is_main_process():
            save_checkpoint(model, optimizer, lr_scheduler, epoch, config)
        
        # 验证
        if (epoch + 1) % config.TRAIN.VAL_INTERVAL == 0:
            val_metrics = validate(model, val_loader, device, config, logger)
            
            # 记录验证指标
            if writer and is_main_process():
                for k, v in val_metrics.items():
                    writer.add_scalar(f'val/{k}', v, epoch)
            
            # 检查是否为最佳模型
            if val_metrics['loss'] < best_val_loss and is_main_process():
                best_val_loss = val_metrics['loss']
                save_checkpoint(model, optimizer, lr_scheduler, epoch, config, is_best=True)
                logger.info(f"发现新的最佳模型，epoch {epoch}, 验证损失: {best_val_loss:.4f}")
            
            # 打印验证结果
            logger.info(
                f"Epoch {epoch} validation: "
                f"Loss: {val_metrics['loss']:.4f}, "
                f"Cls: {val_metrics['cls_loss']:.4f}, "
                f"Reg: {val_metrics['reg_loss']:.4f}, "
                f"Sparsity: {val_metrics['sparsity_loss']:.4f}"
            )
    
    # 保存最终模型
    if is_main_process():
        save_checkpoint(model, optimizer, lr_scheduler, config.TRAIN.EPOCHS-1, config)
    
    logger.info("训练完成!")
    
    # 关闭TensorBoard记录器
    if writer:
        writer.close()


if __name__ == '__main__':
    main()