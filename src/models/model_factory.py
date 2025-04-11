import os
import torch
from typing import Dict, Any, Optional

from src.models.cnn_lstm import CNNLSTM


def create_model(config: Dict[str, Any]) -> torch.nn.Module:
    """
    根据配置创建模型。
    
    参数:
        config: 模型配置字典
        
    返回:
        初始化的模型
        
    异常:
        ValueError: 如果模型类型不支持
    """
    model_type = config.get('type', 'cnn_lstm').lower()
    
    if model_type == 'cnn_lstm':
        model = CNNLSTM.from_config(config)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    # 如果提供了检查点则加载
    if 'checkpoint_path' in config and config['checkpoint_path']:
        load_model_checkpoint(model, config['checkpoint_path'])
    
    return model


def load_model_checkpoint(model: torch.nn.Module, checkpoint_path: str) -> None:
    """
    从检查点文件加载模型权重。
    
    参数:
        model: 要加载权重的模型
        checkpoint_path: 检查点文件路径
        
    异常:
        FileNotFoundError: 如果检查点文件不存在
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"未找到检查点文件: {checkpoint_path}")
    
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    
    # 处理不同的检查点格式
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif isinstance(checkpoint, dict):
        # 如果它看起来只是状态字典，则尝试直接加载
        model.load_state_dict(checkpoint)
    else:
        raise ValueError(f"无法识别{checkpoint_path}中的检查点格式")


def save_model_checkpoint(model: torch.nn.Module, 
                         checkpoint_path: str, 
                         additional_data: Optional[Dict[str, Any]] = None) -> None:
    """
    将模型检查点保存到文件。
    
    参数:
        model: 要保存的模型
        checkpoint_path: 保存检查点的路径
        additional_data: 可选的与模型一起保存的额外数据
    """
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    # 准备检查点数据
    checkpoint = {
        'model_state_dict': model.state_dict(),
    }
    
    # 如果提供了额外数据，则添加
    if additional_data:
        checkpoint.update(additional_data)
    
    # 保存检查点
    torch.save(checkpoint, checkpoint_path) 