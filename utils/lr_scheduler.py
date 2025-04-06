# utils/lr_scheduler.py
import math
import torch
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, StepLR, MultiStepLR

def build_scheduler(config, optimizer, steps_per_epoch=None):
    """
    根据配置构建学习率调度器。
    
    参数:
        config: 包含学习率调度相关配置的对象。
        optimizer: PyTorch优化器。
        steps_per_epoch: 每个epoch的步数（如果需要）。
    
    返回:
        torch.optim.lr_scheduler: 配置好的学习率调度器。
    """
    scheduler_type = config.TRAIN.LR_SCHEDULER
    
    if scheduler_type == 'cosine':
        # 余弦退火学习率调度
        return CosineAnnealingLR(
            optimizer, 
            T_max=config.TRAIN.EPOCHS * steps_per_epoch if steps_per_epoch else config.TRAIN.EPOCHS,
            eta_min=config.TRAIN.MIN_LR
        )
    
    elif scheduler_type == 'step':
        # 阶梯式学习率调度
        return StepLR(
            optimizer,
            step_size=config.TRAIN.LR_STEP_SIZE,
            gamma=config.TRAIN.LR_GAMMA
        )
    
    elif scheduler_type == 'multistep':
        # 多阶梯式学习率调度
        return MultiStepLR(
            optimizer,
            milestones=config.TRAIN.LR_MILESTONES,
            gamma=config.TRAIN.LR_GAMMA
        )
    
    elif scheduler_type == 'warmup_cosine':
        # 带预热的余弦退火
        return WarmupCosineScheduler(
            optimizer,
            warmup_epochs=config.TRAIN.WARMUP_EPOCHS,
            max_epochs=config.TRAIN.EPOCHS,
            warmup_start_lr=config.TRAIN.WARMUP_START_LR,
            eta_min=config.TRAIN.MIN_LR,
            steps_per_epoch=steps_per_epoch
        )
    
    elif scheduler_type == 'warmup_step':
        # 带预热的阶梯式学习率
        return WarmupStepScheduler(
            optimizer,
            warmup_epochs=config.TRAIN.WARMUP_EPOCHS,
            step_size=config.TRAIN.LR_STEP_SIZE,
            gamma=config.TRAIN.LR_GAMMA,
            warmup_start_lr=config.TRAIN.WARMUP_START_LR,
            steps_per_epoch=steps_per_epoch
        )
    
    elif scheduler_type == 'plateau':
        # 根据验证指标调整学习率
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config.TRAIN.LR_GAMMA,
            patience=config.TRAIN.PATIENCE,
            min_lr=config.TRAIN.MIN_LR
        )
    
    else:
        raise ValueError(f"不支持的学习率调度器类型: {scheduler_type}")


class WarmupCosineScheduler(LambdaLR):
    """
    带预热的余弦退火学习率调度器。
    """
    def __init__(
        self,
        optimizer,
        warmup_epochs,
        max_epochs,
        warmup_start_lr=1e-7,
        eta_min=1e-6,
        steps_per_epoch=None,
        last_epoch=-1
    ):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        self.steps_per_epoch = steps_per_epoch
        
        # 计算总的训练步数
        if steps_per_epoch:
            self.warmup_steps = self.warmup_epochs * steps_per_epoch
            self.total_steps = self.max_epochs * steps_per_epoch
        else:
            self.warmup_steps = self.warmup_epochs
            self.total_steps = self.max_epochs
        
        # 保存基础学习率
        self.base_lrs_backup = [group['lr'] for group in optimizer.param_groups]
        
        super(WarmupCosineScheduler, self).__init__(
            optimizer, self.lr_lambda, last_epoch
        )
    
    def lr_lambda(self, step):
        if step < self.warmup_steps:
            # 预热阶段：从warmup_start_lr线性增加到base_lr
            alpha = step / self.warmup_steps
            factor = self.warmup_start_lr / self.base_lrs_backup[0] * (1 - alpha) + alpha
        else:
            # 余弦退火阶段
            factor = self.eta_min / self.base_lrs_backup[0] + 0.5 * (1 - self.eta_min / self.base_lrs_backup[0]) * \
                (1 + math.cos(math.pi * (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)))
        
        return factor


class WarmupStepScheduler(LambdaLR):
    """
    带预热的阶梯式学习率调度器。
    """
    def __init__(
        self,
        optimizer,
        warmup_epochs,
        step_size,
        gamma=0.1,
        warmup_start_lr=1e-7,
        steps_per_epoch=None,
        last_epoch=-1
    ):
        self.warmup_epochs = warmup_epochs
        self.step_size = step_size
        self.gamma = gamma
        self.warmup_start_lr = warmup_start_lr
        self.steps_per_epoch = steps_per_epoch
        
        # 计算总的训练步数
        if steps_per_epoch:
            self.warmup_steps = self.warmup_epochs * steps_per_epoch
            self.decay_steps = self.step_size * steps_per_epoch
        else:
            self.warmup_steps = self.warmup_epochs
            self.decay_steps = self.step_size
        
        # 保存基础学习率
        self.base_lrs_backup = [group['lr'] for group in optimizer.param_groups]
        
        super(WarmupStepScheduler, self).__init__(
            optimizer, self.lr_lambda, last_epoch
        )
    
    def lr_lambda(self, step):
        if step < self.warmup_steps:
            # 预热阶段：从warmup_start_lr线性增加到base_lr
            alpha = step / self.warmup_steps
            factor = self.warmup_start_lr / self.base_lrs_backup[0] * (1 - alpha) + alpha
        else:
            # 阶梯式衰减
            factor = self.gamma ** ((step - self.warmup_steps) // self.decay_steps)
        
        return factor