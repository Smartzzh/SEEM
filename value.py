import os
import torch
import logging
from torch.utils.data import DataLoader
from tqdm import tqdm

# 假设你有一个基础的 Logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class ModelEvaluator:
    def __init__(self, pipeline, opt):
        self.pipeline = pipeline
        self.opt = opt
        self.model_names = None
        self.raw_models = None
        self.device = self.opt['device']
        self.save_folder = self.opt['save_folder']

    def load_model(self, model_path):
        """加载预训练模型"""
        logger.info(f"Loading model from: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        for module_name, model in self.raw_models.items():
            model.load_state_dict(checkpoint.get(module_name))
        logger.info(f"Model loaded successfully from {model_path}")

    def _eval_on_set(self, save_folder):
        """在验证集上进行评估并返回结果"""
        # 假设你有一个验证集 DataLoader
        eval_dataloader = DataLoader(self.pipeline.get_val_dataset(), batch_size=self.opt['batch_size'], num_workers=self.opt['num_workers'], shuffle=False)
        
        self.raw_models.eval()  # 切换到评估模式
        results = {
            'losses': [],
            'metrics': []
        }

        with torch.no_grad():  # 禁用梯度计算
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                images, targets = batch['images'].to(self.device), batch['targets'].to(self.device)

                # 计算模型输出
                outputs = self._forward_pass(images)

                # 计算损失
                loss = self._compute_loss(outputs, targets)
                results['losses'].append(loss.item())

                # 计算评价指标（例如准确率、mIoU等）
                metrics = self._compute_metrics(outputs, targets)
                results['metrics'].append(metrics)

        avg_loss = sum(results['losses']) / len(results['losses'])
        avg_metrics = self._average_metrics(results['metrics'])
        logger.info(f"Validation Loss: {avg_loss}")
        logger.info(f"Validation Metrics: {avg_metrics}")

        return {
            'loss': avg_loss,
            'metrics': avg_metrics
        }

    def _forward_pass(self, images):
        """模型的前向推理"""
        outputs = {}
        for module_name in self.model_names:
            outputs[module_name] = self.raw_models[module_name](images)
        return outputs

    def _compute_loss(self, outputs, targets):
        """计算模型的损失（假设你有一个损失函数）"""
        total_loss = 0
        for module_name, output in outputs.items():
            # 这里的损失计算需要根据具体的任务来编写
            loss = self.pipeline.compute_loss(output, targets[module_name])
            total_loss += loss
        return total_loss

    def _compute_metrics(self, outputs, targets):
        """计算评估指标"""
        # 假设计算了像素精度、IoU等指标
        metrics = {
            'accuracy': self.pipeline.compute_accuracy(outputs, targets),
            'IoU': self.pipeline.compute_IoU(outputs, targets)
        }
        return metrics

    def _average_metrics(self, metrics_list):
        """计算指标的平均值"""
        avg_metrics = {}
        num_metrics = len(metrics_list)
        for metric_name in metrics_list[0].keys():
            avg_metrics[metric_name] = sum(m[metric_name] for m in metrics_list) / num_metrics
        return avg_metrics

    def eval(self):
        """验证模型并返回评估结果"""
        logger.info('-----------------------------------------------')
        logger.info("Evaluating model ... ")
        self.raw_models = self.pipeline.initialize_model()
        self.model_names = self.raw_models.keys()

        # move models to the device
        for module_name in self.model_names:
            self.raw_models[module_name].to(self.device)

        # load model during evaluation
        if self.opt['WEIGHT'] and os.path.isfile(self.opt['RESUME_FROM']):
            model_path = self.opt['RESUME_FROM'] 
            self.load_model(model_path)
        else:
            raise ValueError(f"Model not found: {self.opt['RESUME_FROM']}")

        results = self._eval_on_set(self.save_folder)
        return results

