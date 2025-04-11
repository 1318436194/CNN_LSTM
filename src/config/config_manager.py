import os
import yaml
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """
    从YAML文件加载配置。
    
    参数:
        config_path: 配置文件路径
        
    返回:
        包含配置的字典
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"未找到配置文件: {config_path}")
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    return config


def save_config(config: Dict[str, Any], save_path: str) -> None:
    """
    将配置保存到YAML文件。
    
    参数:
        config: 配置字典
        save_path: 保存配置的路径
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)


def update_config(config: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    使用新值更新配置。
    
    参数:
        config: 原始配置字典
        updates: 包含更新的字典
        
    返回:
        更新后的配置字典
    """
    # 不需要深拷贝，因为我们不修改原始配置
    updated_config = config.copy()
    
    # 递归更新嵌套字典
    def _update_dict(d, u):
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                d[k] = _update_dict(d[k], v)
            else:
                d[k] = v
        return d
    
    return _update_dict(updated_config, updates) 