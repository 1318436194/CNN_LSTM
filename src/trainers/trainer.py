import os
import time
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from typing import Dict, Any, Tuple, List, Optional, Union

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


class EarlyStopping:
    """
    早停机制，防止过拟合。
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, verbose: bool = True):
        """
        初始化早停机制。
        
        参数:
            patience: 验证损失停止改善后等待的轮次数
            min_delta: 验证损失改善的最小变化量
            verbose: 是否打印消息
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        """
        检查是否应该停止训练。
        
        参数:
            val_loss: 当前验证损失
            
        返回:
            如果应该停止训练则为True，否则为False
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f'早停计数器: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


class Trainer:
    """
    时间序列预测的模型训练器。
    """
    
    def __init__(self, 
                model: nn.Module, 
                config: Dict[str, Any], 
                logger: Optional[logging.Logger] = None,
                device: Optional[torch.device] = None):
        """
        初始化训练器。
        
        参数:
            model: 要训练的模型
            config: 训练配置
            logger: 用于记录消息的日志记录器
            device: 运行训练的设备（cpu或cuda）
        """
        self.model = model
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # 确定设备
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.model = self.model.to(self.device)
        
        # 初始化模型权重
        self._initialize_weights()
        
        # 设置优化器
        optimizer_name = config.get('optimizer', 'adam').lower()
        lr = config.get('learning_rate', 0.001)
        
        if optimizer_name == 'adam':
            self.optimizer = optim.Adam(model.parameters(), lr=lr)
        elif optimizer_name == 'sgd':
            self.optimizer = optim.SGD(model.parameters(), lr=lr)
        elif optimizer_name == 'rmsprop':
            self.optimizer = optim.RMSprop(model.parameters(), lr=lr)
        else:
            raise ValueError(f"不支持的优化器: {optimizer_name}")
        
        # 设置损失函数
        loss_name = config.get('loss_function', 'mse').lower()
        
        if loss_name == 'mse':
            self.criterion = nn.MSELoss()
        elif loss_name == 'mae':
            self.criterion = nn.L1Loss()
        else:
            raise ValueError(f"不支持的损失函数: {loss_name}")
        
        # 设置学习率调度器
        if config.get('use_scheduler', False):
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=config.get('scheduler_factor', 0.5),
                patience=config.get('scheduler_patience', 5),
                verbose=True
            )
        else:
            self.scheduler = None
        
        # 如果请求则设置tensorboard
        if config.get('use_tensorboard', False) and SummaryWriter is not None:
            self.writer = SummaryWriter(log_dir=config.get('tensorboard_dir', './logs'))
        else:
            self.writer = None
            
        # 设置梯度裁剪
        self.grad_clip = config.get('grad_clip', 1.0)
    
    def _initialize_weights(self):
        """
        初始化模型权重，以避免训练开始时出现梯度爆炸或消失问题。
        """
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                if len(param.shape) >= 2:  # 对于二维权重矩阵（如线性层和LSTM层）
                    nn.init.xavier_uniform_(param)
                else:  # 对于一维权重（如偏置）
                    nn.init.uniform_(param, -0.1, 0.1)
    
    def train(self, 
             train_loader: torch.utils.data.DataLoader,
             val_loader: torch.utils.data.DataLoader) -> Dict[str, List[float]]:
        """
        训练模型。
        
        参数:
            train_loader: 训练数据的DataLoader
            val_loader: 验证数据的DataLoader
            
        返回:
            包含训练和验证损失的字典
        """
        # 设置早停机制
        early_stopping = EarlyStopping(
            patience=self.config.get('early_stopping_patience', 10),
            verbose=True
        )
        
        # 训练循环
        epochs = self.config.get('epochs', 100)
        history = {'train_loss': [], 'val_loss': []}
        
        self.logger.info(f"在{self.device}上开始训练{epochs}个轮次")
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # 训练阶段
            self.model.train()
            epoch_loss = 0.0
            
            for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                # 前向传播
                self.optimizer.zero_grad()
                forecast_horizon = y_batch.size(1)
                output = self.model(X_batch, forecast_horizon)
                
                # 计算损失
                loss = self.criterion(output, y_batch)
                
                # 检查是否有NaN值
                if torch.isnan(loss).item():
                    self.logger.warning(f"检测到NaN损失，跳过该批次")
                    continue
                
                # 反向传播和优化
                loss.backward()
                
                # 应用梯度裁剪防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            # 计算平均训练损失
            avg_train_loss = epoch_loss / len(train_loader)
            history['train_loss'].append(avg_train_loss)
            
            # 验证阶段
            val_loss = self.evaluate(val_loader)
            history['val_loss'].append(val_loss)
            
            # 如果需要，更新学习率调度器
            if self.scheduler:
                self.scheduler.step(val_loss)
            
            # 记录进度
            elapsed_time = time.time() - start_time
            self.logger.info(
                f"轮次 {epoch+1}/{epochs} - "
                f"训练损失: {avg_train_loss:.6f}, "
                f"验证损失: {val_loss:.6f}, "
                f"时间: {elapsed_time:.2f}秒"
            )
            
            # 如果可用，记录到tensorboard
            if self.writer:
                self.writer.add_scalar('Loss/train', avg_train_loss, epoch)
                self.writer.add_scalar('Loss/validation', val_loss, epoch)
                
                # 记录学习率
                current_lr = self.optimizer.param_groups[0]['lr']
                self.writer.add_scalar('Learning Rate', current_lr, epoch)
            
            # 检查早停
            if early_stopping(val_loss):
                self.logger.info(f"轮次{epoch+1}后触发早停")
                break
        
        self.logger.info("训练完成")
        
        # 如果使用，关闭tensorboard写入器
        if self.writer:
            self.writer.close()
        
        return history
    
    def evaluate(self, data_loader: torch.utils.data.DataLoader) -> float:
        """
        在提供的数据上评估模型。
        
        参数:
            data_loader: 包含评估数据的DataLoader
            
        返回:
            平均损失值
        """
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for X_batch, y_batch in data_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                # 前向传播
                forecast_horizon = y_batch.size(1)
                output = self.model(X_batch, forecast_horizon)
                
                # 计算损失
                loss = self.criterion(output, y_batch)
                total_loss += loss.item()
        
        # 计算平均损失
        avg_loss = total_loss / len(data_loader)
        
        return avg_loss
    
    def predict(self, 
               data_loader: torch.utils.data.DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用模型生成预测。
        
        参数:
            data_loader: 包含输入数据的DataLoader
            
        返回:
            (预测值, 目标值)的元组
        """
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for X_batch, y_batch in data_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                # 前向传播
                forecast_horizon = y_batch.size(1)
                output = self.model(X_batch, forecast_horizon)
                
                # 收集预测值和目标值
                all_predictions.append(output.cpu().numpy())
                all_targets.append(y_batch.cpu().numpy())
        
        # 连接批次
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        
        return predictions, targets
    
    def save_checkpoint(self, 
                       checkpoint_path: str, 
                       additional_data: Optional[Dict[str, Any]] = None) -> None:
        """
        保存模型检查点。
        
        参数:
            checkpoint_path: 保存检查点的路径
            additional_data: 与检查点一起保存的额外数据
        """
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }
        
        if additional_data:
            checkpoint.update(additional_data)
        
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"检查点已保存到 {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        从检查点加载模型。
        
        参数:
            checkpoint_path: 检查点文件的路径
            
        返回:
            包含检查点中额外数据的字典
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"未找到检查点文件: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # 加载模型权重
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # 如果可用，加载优化器状态
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # 返回任何额外数据
        return {k: v for k, v in checkpoint.items() 
                if k not in ['model_state_dict', 'optimizer_state_dict', 'config']}
    
    def compute_metrics(self, 
                       predictions: np.ndarray, 
                       targets: np.ndarray) -> Dict[str, float]:
        """
        计算模型预测的评估指标。
        
        参数:
            predictions: 模型预测值
            targets: 真实值
            
        返回:
            包含评估指标的字典
        """
        # 确保是numpy数组
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        
        # 计算指标
        mse = np.mean((predictions - targets) ** 2)
        mae = np.mean(np.abs(predictions - targets))
        rmse = np.sqrt(mse)
        
        # 可以根据需要添加更多指标
        
        return {
            'mse': float(mse),
            'mae': float(mae),
            'rmse': float(rmse)
        } 