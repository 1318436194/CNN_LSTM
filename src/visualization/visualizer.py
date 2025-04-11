import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple, Optional, Union

# 配置matplotlib支持中文显示
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文黑体
# plt.rcParams['font.sans-serif'] = ['?????']  # 艺术字体
plt.rcParams['axes.unicode_minus'] = False

class Visualizer:
    """
    用于可视化模型结果和预测的类。
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化可视化器。
        
        参数:
            config: 可视化配置
        """
        self.config = config
        self.figsize = config.get('figsize', (10, 6))
        self.dpi = config.get('dpi', 100)
        self.colors = config.get('colors', {
            'prediction': 'red',
            'actual': 'blue'
        })
        self.show_legend = config.get('show_legend', True)
        self.save_format = config.get('save_format', 'png')
    
    def plot_predictions(self, 
                        predictions: np.ndarray, 
                        targets: np.ndarray,
                        save_path: Optional[str] = None,
                        title: str = 'Predictions vs Actual',
                        x_label: str = 'Sample',
                        y_label: str = 'Value') -> None:
        """
        绘制模型预测与实际值的对比图。
        
        参数:
            predictions: 模型预测值
            targets: 实际目标值
            save_path: 保存图表的路径（如果为None，则显示图表）
            title: 图表标题
            x_label: X轴标签
            y_label: Y轴标签
        """
        plt.figure(figsize=self.figsize, dpi=self.dpi)
        
        # 处理不同的维度
        if len(predictions.shape) == 3:
            # 多步预测
            # 获取第一步和最后一步的预测
            first_step_pred = predictions[:, 0, 0]
            last_step_pred = predictions[:, -1, 0]
            first_step_target = targets[:, 0, 0]
            last_step_target = targets[:, -1, 0]
            
            # 绘制第一步
            plt.subplot(2, 1, 1)
            plt.plot(first_step_pred, color=self.colors.get('prediction', 'red'), 
                    label='预测值 (第一步)')
            plt.plot(first_step_target, color=self.colors.get('actual', 'blue'), 
                    label='实际值 (第一步)')
            if self.show_legend:
                plt.legend()
            plt.title(f'第一步 - {title}')
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.grid(True, alpha=0.3)
            
            # 绘制最后一步
            plt.subplot(2, 1, 2)
            plt.plot(last_step_pred, color=self.colors.get('prediction', 'red'), 
                    label='预测值 (最后一步)')
            plt.plot(last_step_target, color=self.colors.get('actual', 'blue'), 
                    label='实际值 (最后一步)')
            if self.show_legend:
                plt.legend()
            plt.title(f'最后一步 - {title}')
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
        else:
            # 单步预测
            plt.plot(predictions, color=self.colors.get('prediction', 'red'), 
                    label='预测值')
            plt.plot(targets, color=self.colors.get('actual', 'blue'), 
                    label='实际值')
            if self.show_legend:
                plt.legend()
            plt.title(title)
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.grid(True, alpha=0.3)
        
        # 保存或显示图表
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, format=self.save_format, dpi=self.dpi)
            plt.close()
        else:
            plt.show()
    
    def plot_forecast(self, 
                     forecast: np.ndarray,
                     save_path: Optional[str] = None,
                     title: str = 'Forecast',
                     x_label: str = 'Future Time Steps',
                     y_label: str = 'Value') -> None:
        """
        绘制预测的时间序列。
        
        参数:
            forecast: 预测值
            save_path: 保存图表的路径（如果为None，则显示图表）
            title: 图表标题
            x_label: X轴标签
            y_label: Y轴标签
        """
        plt.figure(figsize=self.figsize, dpi=self.dpi)
        
        # 处理不同的维度
        if len(forecast.shape) == 3:
            # 多个样本的多步预测
            # 为了可视化，我们将绘制均值和置信区间
            mean_forecast = np.mean(forecast, axis=0)
            std_forecast = np.std(forecast, axis=0)
            upper = mean_forecast + 1.96 * std_forecast
            lower = mean_forecast - 1.96 * std_forecast
            
            # 时间步
            time_steps = np.arange(mean_forecast.shape[0])
            
            # 绘制平均预测
            plt.plot(time_steps, mean_forecast, color=self.colors.get('prediction', 'red'), 
                    label='平均预测')
            
            # 绘制置信区间
            plt.fill_between(time_steps, lower.flatten(), upper.flatten(), 
                           alpha=0.2, color=self.colors.get('prediction', 'red'), 
                           label='95%置信区间')
        else:
            # 单一预测
            plt.plot(forecast, color=self.colors.get('prediction', 'red'), 
                    label='预测')
        
        if self.show_legend:
            plt.legend()
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.grid(True, alpha=0.3)
        
        # 保存或显示图表
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, format=self.save_format, dpi=self.dpi)
            plt.close()
        else:
            plt.show()
    
    def plot_training_history(self, 
                             history: Dict[str, List[float]],
                             save_path: Optional[str] = None,
                             title: str = 'Training History') -> None:
        """
        绘制训练历史（损失曲线）。
        
        参数:
            history: 包含训练历史的字典
            save_path: 保存图表的路径（如果为None，则显示图表）
            title: 图表标题
        """
        plt.figure(figsize=self.figsize, dpi=self.dpi)
        
        # 绘制训练损失
        if 'train_loss' in history:
            plt.plot(history['train_loss'], label='训练损失', color='blue')
        
        # 绘制验证损失
        if 'val_loss' in history:
            plt.plot(history['val_loss'], label='验证损失', color='red')
        
        if self.show_legend:
            plt.legend()
        plt.title(title)
        plt.xlabel('轮次')
        plt.ylabel('损失')
        plt.grid(True, alpha=0.3)
        
        # 保存或显示图表
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, format=self.save_format, dpi=self.dpi)
            plt.close()
        else:
            plt.show() 