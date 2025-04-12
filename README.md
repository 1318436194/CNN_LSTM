# CNN-LSTM 时间序列预测

用于时间序列预测的CNN-LSTM神经网络模块化可扩展框架。本项目提供了一个完整的训练、超参数调优和评估时间序列预测模型的流程。

## 功能特点

- **模块化架构**：清晰分离关注点，数据处理、模型定义、训练、评估和可视化模块相互独立
- **配置驱动**：系统的所有方面都可以通过配置文件控制
- **超参数调优**：使用Optuna内置支持超参数优化
- **可视化**：用于模型预测和训练进度的综合可视化工具
- **TensorBoard集成**：可选集成TensorBoard用于监控训练
- **命令行界面**：用于训练、调优和推理的简单CLI

## 项目结构

```
CNN_LSTM/
├── configs/              # 配置文件
│   └── default_config.yaml  # 默认配置，亦是配置的模版文件
├── datasets/                # 数据集目录
│   └── dataAll.csv		 # 默认数据集，作为Demo
├── weights/             # 预训练权重
├── outputs/             # 默认输出目录
│   └── demo/		     # 在Demo数据集上训练输出的Demo，包括模型权重、训练日志、结果可视化、训练配置，参考训练效果可供参考
├── scripts/             # 常用操作的Shell脚本
│   ├── train.sh         # 训练脚本
│   ├── predict.sh       # 推理预测脚本
│   └── tune.sh          # 超参数调优脚本
├── src/                 # 源代码
│   ├── configs/          # 配置管理
│   ├── data/            # 数据处理
│   ├── models/          # 模型定义
│   ├── trainers/        # 训练和超参数调优
│   ├── utils/           # 工具函数
│   ├── visualization/   # 可视化
│   └── inference/       # 推理预测
├── main.py              # 主程序
├── LICENSE              # 许可
└── README.md            # 项目说明文档
```

## 安装

1. 克隆存储库：

```bash
git clone git@github.com:1318436194/CNN_LSTM.git
cd CNN_LSTM
```

2. 创建虚拟环境并安装依赖：

```bash
conda init
conda config --set auto_activate_base false

# 方法一： 手动配置环境
conda create -n CNN_LSTM -m python=3.10 -y
conda activate CNN_LSTM
pip install torch torchvision torchaudio optuna pandas scikit-learn matplotlib tensorboard
# 50系显卡需要 CUDA 12.8 + Pytorch 2.8
# pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
# pip install optuna pandas scikit-learn matplotlib tensorboard

# 方法二： 从配置文件载入conda环境配置
conda env create -f ./environment.yml

# 方法三： 从配置文件安装Python包
pip install -r ./requirement.txt
```

3. 安装matplotlib可视化中文字体支持：

```bash
./scripts/CN_fonts_support.sh
```

## 使用方法

### 配置

本项目采用YAML文件管理学习配置，您可以修改默认配置文件或参考其创建自己的配置：

```yaml
# 参考 configs/default_config.yaml进行修改
data:
  file_path: "data/your_data.csv"
  # 更多数据配置...

model:
  type: "cnn_lstm"
  input_dim: 2
  # 更多模型配置...

# 更多配置选项...
```

### 训练

使用默认配置训练模型：

```bash
./scripts/train.sh
```

或使用自定义参数动态覆盖指定的配置文件中的参数：

```bash
./scripts/train.sh --config config/your_config.yaml --data_path datasets/your_data.csv --output_dir outputs/your_experiment
```

### 推理

使用训练好的模型生成预测：

```bash
./scripts/predict.sh --checkpoint_path outputs/your_experiment/model_checkpoint.pth --data_path datasets/your_test_data.csv
```

### 超参数调优

在进行训练前，可以使用超参数优化找到更合适的超参数再进行训练：

```bash
./scripts/tune.sh

# 亦可通过命令行动态指定一些参数
# ./scripts/tune.sh --n_trials 50 --output_dir outputs/tuning_results
```

## 自定义

### 添加新的网络模块

添加新模型架构：

1. 在`src/models/`中创建新的模块
2. 实现您的Module类
3. 对`src/models/model_factory.py`中的模型工厂进行更新

### 使用自定义数据集

对于自定义数据格式：

1. 修改`src/data/data_processor.py`中的`TimeSeriesDataProcessor`类
2. 更新配置以匹配您的数据结构

## 许可证

本项目采用MIT许可证 - 详情见LICENSE文件。

## 权重和日志与结果可视化

见 `./logs/*/` 、`./results/*/` 、 `./weights/*/` 文件夹，示范为使用 `./dataAll.csv` 数据集训练和预测的结果。

## 其他

在其他数据集上使用该网络需要相应更改数据处理、网络超参数、可视化方式等。