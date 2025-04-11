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
		self.target_scaler = []
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
		target_columns = self.config.get('target_columns')

		if not feature_columns:
			# 使用除目标列和时间列外的所有列作为特征
			time_column = self.config.get('time_column')
			excluded_columns = [col for col in [target_columns, time_column] if col is not None]
			feature_columns = [col for col in self.df.columns if col not in excluded_columns]

		# 提取特征和目标
		features = self.df[feature_columns].values
		targets = self.df[target_columns].values if target_columns else None

		# 如果启用，应用归一化
		if self.config.get('normalize', True):
			for i, column in enumerate(feature_columns):
				scaler = MinMaxScaler()
				# 对一维数组进行reshape
				col_data = features[:, i].reshape(-1, 1)
				features[:, i] = scaler.fit_transform(col_data).flatten()
				self.feature_scalers[column] = scaler

			if targets is not None:
				for i, column in enumerate(target_columns):
					self.target_scaler.append(MinMaxScaler())
					col_data = targets[:, i].reshape(-1, 1)
					targets[:, i] = self.target_scaler[i].fit_transform(col_data).flatten()

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
		Y = []

		# 创建序列
		for i in range(sequence_length, len(features) - forecast_horizon + 1):
			X.append(features[i - sequence_length:i])

			if targets is not None:
				Y.append(targets[i:i + forecast_horizon])

		X = np.array(X)
		Y = np.array(Y) if targets is not None else None

		# 如果Y是一维的，则进行reshape
		if Y is not None and len(Y.shape) == 2:
			Y = Y.reshape(Y.shape[0], Y.shape[1], len(self.config.get('target_columns', [None])))

		return X, Y

	def split_data(self,
	               X: np.ndarray,
	               y: Optional[np.ndarray] = None) -> Tuple[
		np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
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
		if not self.target_scaler:
			print("警告: 没有找到目标缩放器(target_scaler)，返回未经转换的预测结果")
			return predictions

		# 如需要，将张量转换为numpy
		if isinstance(predictions, torch.Tensor):
			predictions = predictions.detach().cpu().numpy()

		# 输出调试信息
		print(f"逆变换前的预测形状: {predictions.shape}")
		print(f"可用的目标缩放器数量: {len(self.target_scaler)}")
		
		# 保存原始预测值的副本，用于比较
		original_predictions = predictions.copy()
		
		# 处理不同的形状
		original_shape = predictions.shape
		
		try:
			if len(original_shape) == 3:
				# (batch_size, predict_length, feature_nums)
				batch_size, predict_length, feature_nums = original_shape
				
				# 对于单特征情况的特殊处理
				if feature_nums == 1 and len(self.target_scaler) >= 1:
					# 重塑为可处理的形式
					flat_predictions = predictions.reshape(-1, 1)
					# 使用第一个缩放器
					result = self.target_scaler[0].inverse_transform(flat_predictions)
					# 恢复原始形状
					predictions = result.reshape(original_shape)
					print(f"应用了单特征三维预测的逆变换")
				else:
					# 检查target_scaler的长度与feature_nums是否一致
					if len(self.target_scaler) != feature_nums:
						print(f"警告: Target scaler数量({len(self.target_scaler)})与预测特征数量({feature_nums})不匹配")
						# 如果只有一个scaler但有多个特征，使用相同的scaler处理所有特征
						if len(self.target_scaler) == 1:
							print("使用同一个scaler处理所有特征")
							# 重塑为可处理的形式
							predictions_reshaped = predictions.reshape(-1, feature_nums)
							result = np.zeros_like(predictions_reshaped)
							
							for i in range(feature_nums):
								col = predictions_reshaped[:, i].reshape(-1, 1)
								result[:, i] = self.target_scaler[0].inverse_transform(col).flatten()
							
							predictions = result.reshape(original_shape)
						else:
							raise ValueError(f"Target scaler数量({len(self.target_scaler)})与预测特征数量({feature_nums})不匹配")
					else:
						# 标准处理：每个特征使用对应的scaler
						# 重塑为二维数组，便于处理
						predictions_reshaped = predictions.reshape(-1, feature_nums)
						
						# 创建结果数组
						result = np.zeros_like(predictions_reshaped)
						
						# 对每个特征维度单独进行反向变换
						for i in range(feature_nums):
							# 提取当前维度的列
							col = predictions_reshaped[:, i].reshape(-1, 1)
							# 应用反向变换
							result[:, i] = self.target_scaler[i].inverse_transform(col).flatten()
						
						# 恢复原始形状
						predictions = result.reshape(original_shape)
			elif len(original_shape) == 2:
				# (batch_size, feature_nums)
				batch_size, feature_nums = original_shape
				
				# 对于单特征情况的特殊处理
				if feature_nums == 1 and len(self.target_scaler) >= 1:
					# 特殊情况: 单特征预测
					predictions = self.target_scaler[0].inverse_transform(predictions)
					print(f"应用了单特征二维预测的逆变换")
				elif len(self.target_scaler) == 1 and feature_nums > 1:
					# 如果只有一个scaler但有多个特征，使用相同的scaler处理所有特征
					print("使用同一个scaler处理所有特征")
					result = np.zeros_like(predictions)
					
					for i in range(feature_nums):
						col = predictions[:, i].reshape(-1, 1)
						result[:, i] = self.target_scaler[0].inverse_transform(col).flatten()
					
					predictions = result
				elif len(self.target_scaler) != feature_nums:
					raise ValueError(f"Target scaler数量({len(self.target_scaler)})与预测特征数量({feature_nums})不匹配")
				else:
					# 创建结果数组
					result = np.zeros_like(predictions)
					
					# 对每个特征维度单独进行反向变换
					for i in range(feature_nums):
						col = predictions[:, i].reshape(-1, 1)
						result[:, i] = self.target_scaler[i].inverse_transform(col).flatten()
					
					predictions = result
			else:
				# (batch_size, ) 一维数组
				if len(self.target_scaler) >= 1:
					predictions = self.target_scaler[0].inverse_transform(predictions.reshape(-1, 1))
					predictions = predictions.flatten()
					print(f"应用了一维预测的逆变换")
				else:
					raise ValueError(f"没有可用的target_scaler来进行逆变换")
		
			# 输出变换前后的值范围以进行调试
			print(f"变换前的值范围: [{np.min(original_predictions)}, {np.max(original_predictions)}]")
			print(f"变换后的值范围: [{np.min(predictions)}, {np.max(predictions)}]")
			
			return predictions
		except Exception as e:
			print(f"逆变换过程中出错: {str(e)}")
			print("返回未经转换的预测结果")
			return original_predictions

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
			if predictions.shape[1] == 1:
				predictions = predictions.reshape(-1, predictions.shape[-1])
			# 可以添加对其他 3D 情况的处理，但这通常取决于模型的具体输出
			else:
				# 例如，如果输出是 (N, seq_len, features), 可能需要不同的处理方式
				# 这里暂时只处理 (N, 1, 1) 的情况，并对其他3D情况发出警告
				print(
					f"Warning: Unexpected 3D predictions shape {predictions.shape} in save_predictions. Reshaping to (-1, {predictions.shape[-1]}).")
				predictions = predictions.reshape(-1, predictions.shape[-1])  # Fallback reshape
		elif predictions.ndim == 1:
			# 从 (n_samples,) 转换为 (n_samples, 1)
			predictions = predictions.reshape(-1, 1)
		elif predictions.ndim != 2:
			# 对于非 1D/2D/3D 的情况或其他未预料的维度
			raise ValueError(f"Unsupported predictions shape {predictions.shape} for saving.")

		# 获取目标列名，如果不可用则使用默认名称
		target_columns = self.config.get('target_columns', ['prediction'])

		# 创建DataFrame
		df_pred = pd.DataFrame(predictions, columns=target_columns)

		# 保存到CSV
		df_pred.to_csv(save_path, index=False)
		print(f"预测结果已保存到 {save_path}")