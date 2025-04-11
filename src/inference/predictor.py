import torch
import numpy as np
from typing import Dict, Any, Union, Optional

from src.models.model_factory import create_model, load_model_checkpoint


class Predictor:
    """
    用于使用训练好的模型进行预测的类。
    """
    
    def __init__(self, 
                config: Dict[str, Any],
                device: Optional[torch.device] = None):
        """
        初始化预测器。
        
        参数:
            config: 模型配置
            device: 运行推理的设备（cpu或cuda）
        """
        self.config = config
        
        # 确定设备
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # 创建模型
        self.model = create_model(config)
        self.model = self.model.to(self.device)
        
        # 如果提供了checkpoint_path，则加载模型权重
        if 'checkpoint_path' in config and config['checkpoint_path']:
            load_model_checkpoint(self.model, config['checkpoint_path'])
        
        # 将模型设置为评估模式
        self.model.eval()
    
    def predict(self, 
               inputs: Union[torch.Tensor, np.ndarray],
               forecast_horizon: Optional[int] = None) -> np.ndarray:
        """
        使用模型生成预测。
        
        参数:
            inputs: 形状为(batch_size, sequence_length, input_dim)的输入张量或数组
            forecast_horizon: 要预测的时间步数（如果为None，则使用配置中的值）
            
        返回:
            形状为(batch_size, forecast_horizon, output_dim)的预测numpy数组
        """
        # 确定预测视野
        if forecast_horizon is None:
            forecast_horizon = self.config.get('forecast_horizon', 1)
        
        # 如果需要，将numpy数组转换为张量
        if isinstance(inputs, np.ndarray):
            inputs = torch.tensor(inputs, dtype=torch.float32)
        
        # 移至设备
        inputs = inputs.to(self.device)
        
        # 生成预测
        with torch.no_grad():
            predictions = self.model(inputs, forecast_horizon)
        
        # 转换为numpy数组
        predictions = predictions.detach().cpu().numpy()
        
        return predictions
    
    def predict_sequence(self, 
                        initial_input: Union[torch.Tensor, np.ndarray],
                        forecast_horizon: int,
                        use_predictions: bool = True) -> np.ndarray:
        """
        递归地为更长的序列生成预测。
        
        参数:
            initial_input: 形状为(batch_size, sequence_length, input_dim)的初始输入张量或数组
            forecast_horizon: 要预测的总时间步数
            use_predictions: 是否使用预测作为未来步骤的输入
            
        返回:
            形状为(batch_size, forecast_horizon, output_dim)的预测numpy数组
        """
        # 如果需要，将numpy数组转换为张量
        if isinstance(initial_input, np.ndarray):
            initial_input = torch.tensor(initial_input, dtype=torch.float32)
        
        # 移至设备
        initial_input = initial_input.to(self.device)
        
        # 获取维度
        batch_size = initial_input.size(0)
        sequence_length = initial_input.size(1)
        input_dim = initial_input.size(2)
        output_dim = self.config.get('output_dim', 1)
        
        # 初始化输出
        all_predictions = torch.zeros(batch_size, forecast_horizon, output_dim, device=self.device)
        
        # 一次生成一步的预测
        self.model.eval()
        with torch.no_grad():
            current_input = initial_input.clone()
            
            for i in range(forecast_horizon):
                # 生成下一步的预测
                next_step = self.model(current_input, 1)
                all_predictions[:, i:i+1, :] = next_step
                
                if use_predictions and i < forecast_horizon - 1:
                    # 为下一个预测更新输入序列
                    if input_dim == output_dim:
                        # 当输入和输出维度匹配时，我们可以直接使用预测
                        current_input = torch.cat([current_input[:, 1:, :], next_step], dim=1)
                    else:
                        # 当维度不匹配时，我们需要决定如何使用预测
                        # 这只是一个简单的方法 - 您可能需要自定义此方法
                        # 我们假设预测对应于输入的最后一个维度
                        current_input = torch.cat([
                            current_input[:, 1:, :-output_dim],
                            torch.cat([current_input[:, 1:, -output_dim:], next_step], dim=1)
                        ], dim=2)
        
        # 转换为numpy数组
        predictions = all_predictions.detach().cpu().numpy()
        
        return predictions 