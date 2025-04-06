# tool/evaluation.py
import os
import numpy as np
import torch
import json
import time
from tqdm import tqdm
from collections import defaultdict
from nuscenes.eval.detection.config import config_factory
from nuscenes.eval.detection.evaluate import NuScenesEval
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion

class SparseBEVEvaluator:
    """动态稀疏BEV检测器评估工具
    
    用于评估模型在不同光照条件下的性能，以及稀疏化和量化的影响
    """
    
    def __init__(self, cfg, dataset, output_dir=None):
        self.cfg = cfg
        self.dataset = dataset
        self.categories = cfg.DATASET.CLASS_NAMES
        self.output_dir = output_dir or os.path.join(cfg.OUTPUT_DIR, 'eval_results')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 初始化NuScenes评估器
        self.nusc_eval_cfg = config_factory('detection_cvpr_2019')
        
        # 根据光照条件分组的结果
        self.day_results = defaultdict(list)
        self.night_results = defaultdict(list)
        self.twilight_results = defaultdict(list)
        
        # 稀疏性统计
        self.sparsity_stats = {
            'day': [],
            'night': [],
            'twilight': []
        }
        
        # 量化统计
        self.quant_stats = {
            'bit_allocation': defaultdict(int),
            'sensitivity_scores': []
        }
    
    def evaluate(self, model, data_loader, epoch, split='val'):
        """评估模型性能"""
        model.eval()
        results = []
        total_time = 0
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc=f"Evaluating {split}"):
                # 计时开始
                start_time = time.time()
                
                # 模型推理
                outputs = model(batch)
                
                # 计时结束
                inference_time = time.time() - start_time
                total_time += inference_time
                
                # 后处理检测结果
                processed_results = self._post_process(batch, outputs)
                results.extend(processed_results)
                
                # 收集稀疏性统计
                if hasattr(model, 'get_sparsity_stats'):
                    sparsity_info = model.get_sparsity_stats()
                    for sample_token, lighting, sparsity in zip(batch['sample_tokens'], batch['lighting_condition'], sparsity_info):
                        self.sparsity_stats[lighting].append(sparsity)
                
                # 收集量化统计
                if hasattr(model, 'get_quantization_stats') and self.cfg.QUANT.ENABLE:
                    quant_info = model.get_quantization_stats()
                    for bit, count in quant_info['bit_allocation'].items():
                        self.quant_stats['bit_allocation'][bit] += count
                    self.quant_stats['sensitivity_scores'].extend(quant_info['sensitivity_scores'])
        
        # 计算平均推理时间
        avg_inference_time = total_time / len(data_loader)
        
        # 保存结果
        result_path = os.path.join(self.output_dir, f"{split}_results_epoch_{epoch}.json")
        with open(result_path, 'w') as f:
            json.dump(results, f)
        
        # 按光照条件分组结果
        self._group_results_by_lighting(results)
        
        # 计算总体性能指标
        metrics = self._calculate_metrics(results, "overall")
        
        # 计算不同光照条件下的性能指标
        day_metrics = self._calculate_metrics(self.day_results, "day")
        night_metrics = self._calculate_metrics(self.night_results, "night")
        twilight_metrics = self._calculate_metrics(self.twilight_results, "twilight")
        
        # 汇总指标
        summary = {
            'epoch': epoch,
            'inference_time': avg_inference_time,
            'overall': metrics,
            'day': day_metrics,
            'night': night_metrics,
            'twilight': twilight_metrics,
            'sparsity': {
                'day': np.mean(self.sparsity_stats['day']) if self.sparsity_stats['day'] else 0,
                'night': np.mean(self.sparsity_stats['night']) if self.sparsity_stats['night'] else 0,
                'twilight': np.mean(self.sparsity_stats['twilight']) if self.sparsity_stats['twilight'] else 0,
            }
        }
        
        if self.cfg.QUANT.ENABLE:
            summary['quantization'] = {
                'bit_allocation': dict(self.quant_stats['bit_allocation']),
                'avg_sensitivity': np.mean(self.quant_stats['sensitivity_scores']) if self.quant_stats['sensitivity_scores'] else 0
            }
        
        # 保存评估摘要
        summary_path = os.path.join(self.output_dir, f"{split}_summary_epoch_{epoch}.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f)
        
        return summary
    
    def _post_process(self, batch, outputs):
        """将模型输出转换为可评估的格式"""
        results = []
        
        # 解析模型输出
        pred_boxes = outputs['boxes']  # [B, N, 9], 9: (x, y, z, w, l, h, sin, cos, class)
        pred_scores = outputs['scores']  # [B, N]
        pred_classes = outputs['classes']  # [B, N]
        
        for i, sample_token in enumerate(batch['sample_tokens']):
            # 获取当前样本的检测结果
            boxes = pred_boxes[i]
            scores = pred_scores[i]
            classes = pred_classes[i]
            
            # 光照条件
            lighting_condition = batch['lighting_condition'][i]
            
            for j in range(len(boxes)):
                if scores[j] < 0.1:  # 忽略低置信度的检测
                    continue
                
                # 构建检测结果
                detection = {
                    'sample_token': sample_token,
                    'translation': boxes[j, :3].tolist(),
                    'size': boxes[j, 3:6].tolist(),
                    'rotation': [float(boxes[j, 6]), float(boxes[j, 7])],  # [sin, cos]
                    'detection_name': self.categories[int(classes[j])],
                    'detection_score': float(scores[j]),
                    'lighting_condition': lighting_condition
                }
                
                results.append(detection)
        
        return results
    
    def _group_results_by_lighting(self, results):
        """按光照条件分组结果"""
        self.day_results.clear()
        self.night_results.clear()
        self.twilight_results.clear()
        
        for res in results:
            if res['lighting_condition'] == 'day':
                self.day_results[res['sample_token']].append(res)
            elif res['lighting_condition'] == 'night':
                self.night_results[res['sample_token']].append(res)
            elif res['lighting_condition'] == 'twilight':
                self.twilight_results[res['sample_token']].append(res)
    
    def _calculate_metrics(self, results, tag):
        """计算性能指标"""
        if not results:
            return {'mAP': 0, 'mATE': 0, 'mASE': 0, 'mAOE': 0, 'mAVE': 0, 'mAAE': 0, 'NDS': 0}
        
        # 转换为NuScenes评估格式
        nusc_results = []
        for sample_token, detections in results.items():
            for detection in detections:
                # 转换为NuScenes Box格式
                box = Box(
                    center=detection['translation'],
                    size=detection['size'],
                    orientation=Quaternion(axis=[0, 0, 1], angle=np.arctan2(detection['rotation'][0], detection['rotation'][1])),
                    label=detection['detection_name'],
                    score=detection['detection_score']
                )
                
                nusc_results.append({
                    'sample_token': sample_token,
                    'translation': box.center.tolist(),
                    'size': box.wlh.tolist(),
                    'rotation': box.orientation.elements.tolist(),
                    'detection_name': detection['detection_name'],
                    'detection_score': detection['detection_score'],
                    'attribute_name': ''  # 暂不使用属性
                })
        
        # 保存评估结果
        result_path = os.path.join(self.output_dir, f"{tag}_nusc_results.json")
        with open(result_path, 'w') as f:
            json.dump(nusc_results, f)
        
        # 使用NuScenes评估器计算指标
        try:
            nusc_eval = NuScenesEval(
                self.dataset.nusc,
                config=self.nusc_eval_cfg,
                result_path=result_path,
                eval_set=self.dataset.split,
                output_dir=os.path.join(self.output_dir, tag),
                verbose=False
            )
            metrics = nusc_eval.main()
        except Exception as e:
            print(f"NuScenes评估出错: {e}")
            metrics = {'mAP': 0, 'mATE': 0, 'mASE': 0, 'mAOE': 0, 'mAVE': 0, 'mAAE': 0, 'NDS': 0}
        
        return metrics
    
    def analyze_sparsity_impact(self):
        """分析稀疏性对性能的影响"""
        if not any(self.sparsity_stats.values()):
            return None
        
        # 计算不同稀疏度下的性能
        sparsity_metrics = {}
        
        # 将稀疏度分成几个区间
        sparsity_bins = np.linspace(0, 1, 5)
        
        for condition in ['day', 'night', 'twilight']:
            sparsity_metrics[condition] = []
            
            # 按区间分组
            for i in range(len(sparsity_bins) - 1):
                low, high = sparsity_bins[i], sparsity_bins[i+1]
                bin_samples = [s for s in self.sparsity_stats[condition] if low <= s < high]
                
                if bin_samples:
                    avg_sparsity = np.mean(bin_samples)
                    # 这里可以计算该区间的性能指标，或者使用预先计算的性能
                    sparsity_metrics[condition].append({
                        'sparsity_range': [float(low), float(high)],
                        'avg_sparsity': float(avg_sparsity),
                        'sample_count': len(bin_samples)
                    })
        
        return sparsity_metrics
    
    def analyze_quantization_impact(self):
        """分析量化对性能的影响"""
        if not self.cfg.QUANT.ENABLE or not self.quant_stats['bit_allocation']:
            return None
        
        # 计算不同位宽的分布
        total_layers = sum(self.quant_stats['bit_allocation'].values())
        bit_distribution = {
            str(bit): count / total_layers
            for bit, count in self.quant_stats['bit_allocation'].items()
        }
        
        # 敏感度分析
        sensitivity_distribution = {}
        if self.quant_stats['sensitivity_scores']:
            sensitivity_scores = np.array(self.quant_stats['sensitivity_scores'])
            
            # 计算敏感度分布
            hist, bins = np.histogram(sensitivity_scores, bins=10, range=(0, 1))
            sensitivity_distribution = {
                f"{bins[i]:.2f}-{bins[i+1]:.2f}": int(hist[i])
                for i in range(len(hist))
            }
        
        return {
            'bit_distribution': bit_distribution,
            'sensitivity_distribution': sensitivity_distribution,
            'avg_sensitivity': float(np.mean(self.quant_stats['sensitivity_scores'])) if self.quant_stats['sensitivity_scores'] else 0
        }