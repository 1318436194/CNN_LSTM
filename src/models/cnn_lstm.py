import torch
import torch.nn as nn
from typing import Dict, Any, Tuple


class CNNLSTM(nn.Module):
    """
    结合CNN和LSTM的时间序列预测模型。
    
    该模型首先应用一维卷积从输入序列中提取特征，
    然后使用LSTM编码器对序列进行编码。最后，使用LSTM解码器
    生成输出序列。
    """
    
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int, 
                 output_dim: int, 
                 num_layers: int = 2, 
                 dropout: float = 0.2,
                 kernel_size: int = 3):
        """
        初始化CNN-LSTM模型。
        
        参数:
            input_dim: 输入特征数量
            hidden_dim: LSTM的隐藏维度大小
            output_dim: 输出特征数量
            num_layers: LSTM层数
            dropout: Dropout率
            kernel_size: CNN层的核大小
        """
        super(CNNLSTM, self).__init__()
        
        # 用于特征提取的CNN层
        self.cnn = nn.Conv1d(
            in_channels=input_dim, 
            out_channels=input_dim, 
            kernel_size=kernel_size, 
            padding=kernel_size//2
        )
        
        # LSTM编码器
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # LSTM解码器
        self.decoder = nn.LSTM(
            input_size=output_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # 输出层
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x: torch.Tensor, target_len: int = 1) -> torch.Tensor:
        """
        模型的前向传播。
        
        参数:
            x: 形状为(batch_size, seq_len, input_dim)的输入张量
            target_len: 要生成的目标序列长度
            
        返回:
            形状为(batch_size, target_len, output_dim)的输出张量
        """
        batch_size = x.size(0)
        
        # 应用CNN进行特征提取
        # 为CNN调整维度顺序: (batch_size, input_dim, seq_len)
        x_cnn = x.permute(0, 2, 1)
        x_cnn = torch.relu(self.cnn(x_cnn))
        # 维度顺序调整回来: (batch_size, seq_len, input_dim)
        x_cnn = x_cnn.permute(0, 2, 1)
        
        # 编码输入序列
        _, (hidden, cell) = self.encoder(x_cnn)
        
        # 用零初始化解码器输入
        decoder_input = torch.zeros(batch_size, target_len, self.fc.out_features, device=x.device)
        
        # 使用解码器生成预测
        decoder_output, _ = self.decoder(decoder_input, (hidden, cell))
        
        # 应用输出层
        output = self.fc(decoder_output)
        
        return output
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'CNNLSTM':
        """
        从配置创建模型实例。
        
        参数:
            config: 模型配置字典
            
        返回:
            初始化的CNNLSTM模型
        """
        return cls(
            input_dim=config['input_dim'],
            hidden_dim=config['hidden_dim'],
            output_dim=config['output_dim'],
            num_layers=config['num_layers'],
            dropout=config['dropout'],
            kernel_size=config.get('kernel_size', 3)
        ) 