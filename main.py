import argparse
import os
import yaml
import logging

from src.config.config_manager import load_config
from src.data.data_processor import TimeSeriesDataProcessor
from src.models.model_factory import create_model
from src.trainers.trainer import Trainer
from src.trainers.hyperparameter_tuning import tune_hyperparameters
from src.inference.predictor import Predictor
from src.visualization.visualizer import Visualizer
from src.utils.logger import setup_logger


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description='CNN-LSTM 时间序列预测')
    
    # 主要参数
    parser.add_argument('--mode', type=str, choices=['train', 'predict', 'tune'],
                        default='train', help='运行模式: train(训练), predict(预测), 或 tune(调优)')
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                        help='配置文件路径')
    
    # 数据参数
    parser.add_argument('--data_path', type=str, default=None,
                        help='输入数据文件路径')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='保存输出的目录')
    
    # 模型参数
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='用于预测或微调的模型检查点路径')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=None,
                        help='训练轮次数')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='训练的批量大小')
    parser.add_argument('--learning_rate', type=float, default=None,
                        help='优化器学习率')
    
    # 调优参数
    parser.add_argument('--n_trials', type=int, default=None,
                        help='超参数调优的试验次数')
    
    # 预测参数
    parser.add_argument('--forecast_horizon', type=int, default=None,
                        help='预测的时间步数')
    
    return parser.parse_args()


def update_config_from_args(config, args):
    """使用命令行参数更新配置。"""
    # 更新数据配置
    if args.data_path:
        config['data']['file_path'] = args.data_path
    if args.output_dir:
        config['output_dir'] = args.output_dir
    
    # 更新模型配置
    if args.checkpoint_path:
        config['model']['checkpoint_path'] = args.checkpoint_path
    
    # 更新训练配置
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['data']['batch_size'] = args.batch_size
    if args.learning_rate:
        config['model']['learning_rate'] = args.learning_rate
    
    # 更新调优配置
    if args.n_trials:
        config['tuning']['n_trials'] = args.n_trials
    
    # 更新预测配置
    if args.forecast_horizon:
        config['data']['forecast_horizon'] = args.forecast_horizon
    
    return config


def train(config, logger):
    """使用给定配置训练模型。"""
    logger.info("开始训练")
    
    # 初始化数据处理器并准备数据
    data_processor = TimeSeriesDataProcessor(config['data'])
    train_loader, val_loader, test_loader, scalers = data_processor.prepare_data_for_training()
    
    # 创建模型
    model = create_model(config['model'])
    
    # 初始化训练器
    trainer = Trainer(model, config['training'], logger)
    
    # 训练模型
    history = trainer.train(train_loader, val_loader)
    
    # 在测试集上评估
    test_loss = trainer.evaluate(test_loader)
    logger.info(f"测试损失: {test_loss:.6f}")
    
    # 在测试集上生成预测
    predictions, targets = trainer.predict(test_loader)
    
    # 如果需要，将预测值和目标值反向变换
    if 'target_scaler' in scalers and scalers['target_scaler'] is not None:
        predictions = data_processor.inverse_transform_target(predictions)
        targets = data_processor.inverse_transform_target(targets)
    
    # 计算指标
    metrics = trainer.compute_metrics(predictions, targets)
    logger.info(f"测试指标: {metrics}")
    
    # 保存模型检查点
    output_dir = config.get('output_dir', 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, 'model_checkpoint.pth')
    trainer.save_checkpoint(checkpoint_path, {'metrics': metrics, 'config': config})
    
    # 可视化结果
    visualizer = Visualizer(config.get('visualization', {}))
    
    # 绘制预测图
    predictions_plot_path = os.path.join(output_dir, 'predictions.png')
    visualizer.plot_predictions(predictions, targets, predictions_plot_path)
    
    # 绘制训练历史图
    history_plot_path = os.path.join(output_dir, 'training_history.png')
    visualizer.plot_training_history(history, history_plot_path)
    
    logger.info(f"训练完成。模型已保存到 {checkpoint_path}")
    return metrics


def predict(config, logger):
    """使用训练好的模型生成预测。"""
    logger.info("开始预测")
    
    # 检查是否提供了检查点路径
    if not config['model'].get('checkpoint_path'):
        logger.error("未提供用于预测的检查点路径")
        return
    
    # 初始化数据处理器并准备数据
    data_processor = TimeSeriesDataProcessor(config['data'])
    input_data = data_processor.prepare_data_for_inference()
    
    # 初始化预测器
    predictor = Predictor(config['model'])
    
    # 生成预测
    forecast_horizon = config['data'].get('forecast_horizon', 1)
    predictions = predictor.predict(input_data, forecast_horizon)
    
    # 如果需要，反向变换预测值
    if hasattr(data_processor, 'target_scaler') and data_processor.target_scaler is not None:
        predictions = data_processor.inverse_transform_target(predictions)
    
    # 保存预测结果
    output_dir = config.get('output_dir', 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    predictions_path = os.path.join(output_dir, 'predictions.csv')
    data_processor.save_predictions(predictions, predictions_path)
    
    # 可视化预测结果
    visualizer = Visualizer(config.get('visualization', {}))
    plot_path = os.path.join(output_dir, 'forecast.png')
    visualizer.plot_forecast(predictions, plot_path)
    
    logger.info(f"预测完成。结果已保存到 {output_dir}")
    return predictions


def tune(config, logger):
    """为模型调优超参数。"""
    logger.info("开始超参数调优")

    # 禁止从检查点加载模型
    config['model']['use_checkpoint'] = False
    
    # 初始化数据处理器并准备数据
    data_processor = TimeSeriesDataProcessor(config['data'])
    train_loader, val_loader, _, _ = data_processor.prepare_data_for_training()
    
    # 调优超参数
    best_params = tune_hyperparameters(
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        trainer_class=Trainer,
        logger=logger
    )
    
    # 使用最佳参数更新模型配置
    for key, value in best_params.items():
        config['model'][key] = value
    
    # 保存更新后的配置
    output_dir = config.get('output_dir', 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join(output_dir, 'tuned_config.yaml')
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logger.info(f"超参数调优完成。最佳参数: {best_params}")
    logger.info(f"更新后的配置已保存到 {config_path}")
    
    return best_params


def main():
    """主入口点。"""
    # 解析命令行参数
    args = parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 使用命令行参数更新配置
    config = update_config_from_args(config, args)
    
    # 设置日志记录
    output_dir = config.get('output_dir', 'outputs')
    logger = setup_logger(output_dir)
    
    # 以选定的模式运行
    if args.mode == 'train':
        train(config, logger)
    elif args.mode == 'predict':
        predict(config, logger)
    elif args.mode == 'tune':
        tune(config, logger)


if __name__ == '__main__':
    main() 