import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from typing import Dict, Any, Tuple, List, Union, Optional


class TimeSeriesDataProcessor:
    """
    用于时间序列预测任务的数据处理类。
    
    该类处理数据加载、预处理、创建序列
    以及将数据分割为训练集、验证集和测试集。
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化数据处理器。
        
        参数:
            config: 数据配置字典
        """
        self.config = config
        self.feature_scalers = {}
        self.target_scaler = None
        self.df = None
    
    def load_data(self) -> pd.DataFrame:
        """
        从配置中指定的文件加载数据。
        
        返回:
            包含加载数据的DataFrame
        """
        file_path = self.config['file_path']
        time_column = self.config.get('time_column')
        
        # 加载数据
        df = pd.read_csv(file_path)
        
        # 如果指定了时间列，则按时间列排序
        if time_column and time_column in df.columns:
            df = df.sort_values(by=time_column, ascending=True)
        
        self.df = df
        return df
    
    def preprocess_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        预处理加载的数据，包括归一化和特征提取。
        
        返回:
            (特征, 目标)的numpy数组元组
        """
        if self.df is None:
            self.load_data()
        
        # 提取特征列和目标列
        feature_columns = self.config.get('feature_columns', [])
        target_column = self.config.get('target_column')
        
        if not feature_columns:
            # 使用除目标列和时间列外的所有列作为特征
            time_column = self.config.get('time_column')
            excluded_columns = [col for col in [target_column, time_column] if col is not None]
            feature_columns = [col for col in self.df.columns if col not in excluded_columns]
        
        # 提取特征和目标
        features = self.df[feature_columns].values
        targets = self.df[target_column].values if target_column else None
        
        # 如果启用，应用归一化
        if self.config.get('normalize', True):
            for i, column in enumerate(feature_columns):
                scaler = MinMaxScaler()
                # 对一维数组进行reshape
                col_data = features[:, i].reshape(-1, 1)
                features[:, i] = scaler.fit_transform(col_data).flatten()
                self.feature_scalers[column] = scaler
            
            if targets is not None:
                self.target_scaler = MinMaxScaler()
                targets = self.target_scaler.fit_transform(targets.reshape(-1, 1)).flatten()
        
        return features, targets
    
    def create_sequences(self, 
                         features: np.ndarray, 
                         targets: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        为训练创建输入序列和目标序列。
        
        参数:
            features: 特征数组
            targets: 目标数组（可选）
            
        返回:
            (输入序列, 目标序列)的元组
        """
        sequence_length = self.config.get('sequence_length', 30)
        forecast_horizon = self.config.get('forecast_horizon', 1)
        
        X = []
        y = []
        
        # 创建序列
        for i in range(sequence_length, len(features) - forecast_horizon + 1):
            X.append(features[i - sequence_length:i])
            
            if targets is not None:
                y.append(targets[i:i + forecast_horizon])
        
        X = np.array(X)
        y = np.array(y) if targets is not None else None
        
        # 如果y是一维的，则进行reshape
        if y is not None and len(y.shape) == 2:
            y = y.reshape(y.shape[0], y.shape[1], 1)
        
        return X, y
    
    def split_data(self, 
                  X: np.ndarray, 
                  y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        将数据分割为训练集、验证集和测试集。
        
        参数:
            X: 输入序列
            y: 目标序列（可选）
            
        返回:
            (X_train, X_val, X_test, y_train, y_val, y_test)的元组
        """
        train_ratio = self.config.get('train_ratio', 0.6)
        val_ratio = self.config.get('val_ratio', 0.2)
        
        # 计算分割索引
        train_size = int(len(X) * train_ratio)
        val_size = int(len(X) * val_ratio)
        
        # 分割X
        X_train = X[:train_size]
        X_val = X[train_size:train_size + val_size]
        X_test = X[train_size + val_size:]
        
        # 如果提供了y，则分割y
        if y is not None:
            y_train = y[:train_size]
            y_val = y[train_size:train_size + val_size]
            y_test = y[train_size + val_size:]
        else:
            y_train, y_val, y_test = None, None, None
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def prepare_data_for_training(self) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Any]]:
        """
        准备用于模型训练的数据，包括加载、预处理、
        创建序列和分割为数据加载器。
        
        返回:
            (train_loader, val_loader, test_loader, scalers)的元组
        """
        # 加载并预处理数据
        features, targets = self.preprocess_data()
        
        # 创建序列
        X, y = self.create_sequences(features, targets)
        
        # 分割数据
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)
        
        # 转换为张量
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
        
        # 创建数据加载器
        batch_size = self.config.get('batch_size', 64)
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=self.config.get('shuffle', True)
        )
        
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        # 准备用于后续使用的缩放器字典
        scalers = {
            'feature_scalers': self.feature_scalers,
            'target_scaler': self.target_scaler
        }
        
        return train_loader, val_loader, test_loader, scalers
    
    def prepare_data_for_inference(self) -> torch.Tensor:
        """
        准备用于模型推理的数据。
        
        返回:
            准备好进行推理的输入序列张量
        """
        # 加载并预处理数据
        features, _ = self.preprocess_data()
        
        # 创建序列（无目标）
        X, _ = self.create_sequences(features)
        
        # 转换为张量
        X_tensor = torch.tensor(X, dtype=torch.float32)
        
        return X_tensor
    
    def inverse_transform_target(self, predictions: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        将缩放后的预测值反向变换回原始尺度。
        
        参数:
            predictions: 模型预测值（numpy数组或张量）
            
        返回:
            原始尺度的预测值
        """
        if self.target_scaler is None:
            return predictions
        
        # 如需要，将张量转换为numpy
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        
        # 处理不同的形状
        original_shape = predictions.shape
        if len(original_shape) == 3:
            # (batch_size, sequence_length, 1)
            predictions = predictions.reshape(-1, 1)
            predictions = self.target_scaler.inverse_transform(predictions)
            predictions = predictions.reshape(original_shape)
        elif len(original_shape) == 2:
            # (batch_size, 1)
            predictions = self.target_scaler.inverse_transform(predictions)
        else:
            # (batch_size, )
            predictions = self.target_scaler.inverse_transform(predictions.reshape(-1, 1))
            predictions = predictions.flatten()
        
        return predictions
    
    def save_predictions(self, predictions: np.ndarray, save_path: str) -> None:
        """
        将预测值保存到CSV文件。
        
        参数:
            predictions: 要保存的预测值
            save_path: 保存预测值的路径
        """
        # 确保存储前 predictions 是 2D numpy 数组 [n_samples, n_features]
        if predictions.ndim == 3:
            # 假设形状为 (n_samples, 1, n_features) 或 (n_samples, sequence_length, 1)
            # 我们需要将其转换为 (n_samples, n_features) 或 (n_samples * sequence_length, 1)
            # 根据错误信息 (N, 1, 1)，最可能的情况是 (n_samples, 1, 1) -> (n_samples, 1)
            if predictions.shape[1] == 1 and predictions.shape[2] == 1:
                predictions = predictions.reshape(-1, 1)
            # 可以添加对其他 3D 情况的处理，但这通常取决于模型的具体输出
            else:
                 # 例如，如果输出是 (N, seq_len, features), 可能需要不同的处理方式
                 # 这里暂时只处理 (N, 1, 1) 的情况，并对其他3D情况发出警告
                 print(f"Warning: Unexpected 3D predictions shape {predictions.shape} in save_predictions. Reshaping to (-1, 1).")
                 predictions = predictions.reshape(-1, 1) # Fallback reshape
        elif predictions.ndim == 1:
            # 从 (n_samples,) 转换为 (n_samples, 1)
            predictions = predictions.reshape(-1, 1)
        elif predictions.ndim != 2:
            # 对于非 1D/2D/3D 的情况或其他未预料的维度
             raise ValueError(f"Unsupported predictions shape {predictions.shape} for saving.")

        # 获取目标列名，如果不可用则使用默认名称
        target_column = self.config.get('target_column', 'prediction')
        
        # 如果有多列预测结果，生成列名
        num_features = predictions.shape[1]
        if num_features > 1:
            columns = [f"{target_column}_{i}" for i in range(num_features)]
        else:
            columns = [target_column]

        # 创建DataFrame
        df_pred = pd.DataFrame(predictions, columns=columns)
        
        # 保存到CSV
        df_pred.to_csv(save_path, index=False)
        print(f"预测结果已保存到 {save_path}")