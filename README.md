## 环境配置

```shell
conda init
conda config --set auto_activate_base false

# 方法一： 手动配置环境
conda create -n CNN_LSTM -m python=3.10
codna activate CNN_LSTM
pip install torch torchvision torchaudio optuna pandas scikit-learn matplotlib
# 50系显卡需要 CUDA 12.8 + Pytorch 2.8
# pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
# pip install optuna pandas scikit-learn matplotlib

# 方法二： 从配置文件载入环境配置
conda env create -f environment.yml
```

## weights & logs & results visualization

见 `./*/logs` 、`./*/results` 、 `./*/weights` 文件夹，示范为使用 `./dataAll.csv` 数据集训练和预测的结果。

## 其他

在其他数据集上使用该网络需要相应更改数据处理、网络超参数、可视化方式等。