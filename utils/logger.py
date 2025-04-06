# utils/logger.py
import os
import logging
import time
import datetime
import torch

def setup_logger(name, log_file=None, distributed_rank=0):
    """
    设置日志记录器。
    
    参数:
        name (str): 日志记录器名称。
        log_file (str, optional): 日志文件路径。如果为None，则只输出到控制台。
        distributed_rank (int, optional): 在分布式训练中的rank。只有rank=0的进程会记录日志。
    
    返回:
        logging.Logger: 配置好的日志记录器。
    """
    # 只允许主进程记录日志
    if distributed_rank > 0:
        return logging.getLogger(name)
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    
    # 清除已有的处理器
    for handler in logger.handlers:
        logger.removeHandler(handler)
    
    # 定义格式化器
    fmt = '[%(asctime)s] %(levelname)s: %(message)s'
    color_fmt = '\033[36m[%(asctime)s]\033[0m \033[32m%(levelname)s\033[0m: %(message)s'
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(console_handler)
    
    # 文件处理器（如果指定了日志文件）
    if log_file is not None:
        # 确保日志目录存在
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
            
        # 添加文件处理器
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(file_handler)
    
    return logger

class MetricLogger:
    """
    用于记录和打印训练/验证指标的辅助类。
    """
    def __init__(self, delimiter=", "):
        self.meters = {}
        self.delimiter = delimiter
        self.start_time = time.time()
    
    def add_meter(self, name, meter):
        self.meters[name] = meter
    
    def add_scalar(self, name, value, n=1):
        if name not in self.meters:
            self.meters[name] = SmoothedValue(window_size=20)
        self.meters[name].update(value, n)
    
    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.add_scalar(k, v)
    
    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        return getattr(self, attr)
    
    def log_every(self, logger, iterable, print_freq, header=None):
        i = 0
        if header is not None:
            logger.info(header)
        
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(window_size=20)
        data_time = SmoothedValue(window_size=20)
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                logger.info(
                    header + self.delimiter +
                    "[{" + space_fmt + "}/{" + space_fmt + "}]" + self.delimiter +
                    "eta: {}" + self.delimiter +
                    "{:.4f}s/it" + self.delimiter +
                    "{}".format(
                        i, len(iterable), eta_string, iter_time.global_avg,
                        self.delimiter.join(str(meter) for meter in self.meters.values())
                    )
                )
            
            i += 1
            end = time.time()
        
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logger.info("{} 总时间: {} ({:.4f}s/it)".format(
            header, total_time_str, total_time / len(iterable)))

class SmoothedValue:
    """
    用于跟踪一系列值并计算其均值和中位数的类。
    """
    def __init__(self, window_size=20):
        self.window_size = window_size
        self.reset()
    
    def reset(self):
        self.values = []
        self.counts = []
        self.sum = 0.0
        self.count = 0
    
    def update(self, value, n=1):
        self.values.append(value)
        self.counts.append(n)
        if len(self.values) > self.window_size:
            self.sum -= self.values.pop(0) * self.counts.pop(0)
            self.count -= self.counts[0]
        self.sum += value * n
        self.count += n
    
    @property
    def median(self):
        import numpy as np
        return np.median(self.values)
    
    @property
    def avg(self):
        return sum(self.values) / max(1, len(self.values))
    
    @property
    def global_avg(self):
        return self.sum / max(1, self.count)
    
    def __str__(self):
        return "{median:.4f} ({global_avg:.4f})".format(
            median=self.median, global_avg=self.global_avg)