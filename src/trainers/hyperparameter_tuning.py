import optuna
import torch
import logging
from typing import Dict, Any, Optional, Callable, List, Union

from src.models.model_factory import create_model


def objective_factory(
    base_config: Dict[str, Any],
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    trainer_class: Callable,
    logger: Optional[logging.Logger] = None
) -> Callable:
    """
    为Optuna超参数调优创建目标函数。
    
    参数:
        base_config: 基础模型配置
        train_loader: 训练数据的DataLoader
        val_loader: 验证数据的DataLoader
        trainer_class: 要使用的训练器类
        logger: 用于记录消息的日志记录器
        
    返回:
        用于Optuna的目标函数
    """
    
    def objective(trial: optuna.Trial) -> float:
        """
        超参数优化的目标函数。
        
        参数:
            trial: Optuna试验对象
            
        返回:
            验证损失
        """
        # 创建基础配置的副本
        config = base_config.copy()
        
        # 定义要调优的超参数
        if 'hidden_dim' in config.get('tuning', {}).get('parameters', {}):
            hidden_dim_params = config['tuning']['parameters']['hidden_dim']
            config['model']['hidden_dim'] = trial.suggest_int(
                'hidden_dim', 
                hidden_dim_params.get('min', 50), 
                hidden_dim_params.get('max', 200)
            )
        
        if 'num_layers' in config.get('tuning', {}).get('parameters', {}):
            num_layers_params = config['tuning']['parameters']['num_layers']
            config['model']['num_layers'] = trial.suggest_int(
                'num_layers', 
                num_layers_params.get('min', 1), 
                num_layers_params.get('max', 3)
            )
        
        if 'dropout' in config.get('tuning', {}).get('parameters', {}):
            dropout_params = config['tuning']['parameters']['dropout']
            config['model']['dropout'] = trial.suggest_float(
                'dropout', 
                dropout_params.get('min', 0.1), 
                dropout_params.get('max', 0.5)
            )
        
        if 'learning_rate' in config.get('tuning', {}).get('parameters', {}):
            lr_params = config['tuning']['parameters']['learning_rate']
            if lr_params.get('log', False):
                config['model']['learning_rate'] = trial.suggest_float(
                    'learning_rate', 
                    lr_params.get('min', 0.0001), 
                    lr_params.get('max', 0.1),
                    log=True
                )
            else:
                config['model']['learning_rate'] = trial.suggest_float(
                    'learning_rate', 
                    lr_params.get('min', 0.0001), 
                    lr_params.get('max', 0.1)
                )
        
        # 可以在此处添加更多超参数
        
        if logger:
            logger.info(f"试验 {trial.number}: {config['model']}")
        
        # 训练几个轮次
        num_epochs = config.get('tuning', {}).get('epochs_per_trial', 10)
        config['model']['epochs'] = num_epochs

        # 使用试验的超参数创建模型
        model = create_model(config['model'])

        # 创建训练器
        trainer = trainer_class(
            model=model,
            config=config['model'],
            logger=logger
        )
        
        # 训练和评估
        trainer.train(train_loader, val_loader)
        
        # 获取最佳验证损失
        val_loss = trainer.evaluate(val_loader)
        
        return val_loss
    
    return objective


def tune_hyperparameters(
    config: Dict[str, Any],
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    trainer_class: Callable,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    使用Optuna调优模型超参数。
    
    参数:
        config: 模型和训练配置
        train_loader: 训练数据的DataLoader
        val_loader: 验证数据的DataLoader
        trainer_class: 要使用的训练器类
        logger: 用于记录消息的日志记录器
        
    返回:
        包含最佳超参数的字典
    """
    n_trials = config.get('tuning', {}).get('n_trials', 10)
    
    if logger:
        logger.info(f"开始超参数调优，共{n_trials}次试验")
    
    # 创建研究对象
    study = optuna.create_study(direction="minimize", study_name=f"CNN_LSTM超参数调优试验")
    
    # 创建目标函数
    objective = objective_factory(
        base_config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        trainer_class=trainer_class,
        logger=logger
    )
    
    # 优化
    study.optimize(objective, n_trials=n_trials)
    
    # 获取最佳参数
    best_trial = study.best_trial
    best_params = best_trial.params
    best_value = best_trial.value
    
    if logger:
        logger.info(f"最佳试验: {best_trial.number}")
        logger.info(f"最佳参数: {best_params}")
        logger.info(f"最佳验证损失: {best_value:.6f}")
    
    return best_params 