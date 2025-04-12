#! /bin/bash

# 使用训练好的CNN-LSTM模型生成预测

# 默认配置文件
CONFIG_FILE="configs/SCY_config.yaml"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
  case $1 in
    --config)
      CONFIG_FILE="$2"
      shift 2
      ;;
    --data_path)
      DATA_PATH="$2"
      shift 2
      ;;
    --checkpoint_path)
      CHECKPOINT_PATH="$2"
      shift 2
      ;;
    --output_dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --forecast_horizon)
      FORECAST_HORIZON="$2"
      shift 2
      ;;
    *)
      # 未知选项
      echo "未知选项: $1"
      exit 1
      ;;
  esac
done

# 检查是否提供了检查点路径
# if [ -z "$CHECKPOINT_PATH" ]; then
#   echo "错误: 预测需要提供--checkpoint_path参数"
#   exit 1
# fi

# 构建命令
CMD="python main.py --mode predict --config ${CONFIG_FILE}"

# 如果提供了可选参数，则添加
if [ ! -z "$CHECKPOINT_PATH" ]; then
  CMD="$CMD --checkpoint_path $CHECKPOINT_PATH"
fi

if [ ! -z "$DATA_PATH" ]; then
  CMD="$CMD --data_path $DATA_PATH"
fi

if [ ! -z "$OUTPUT_DIR" ]; then
  CMD="$CMD --output_dir $OUTPUT_DIR"
fi

if [ ! -z "$FORECAST_HORIZON" ]; then
  CMD="$CMD --forecast_horizon $FORECAST_HORIZON"
fi

# 打印命令
echo "执行: $CMD"

# 执行命令
eval $CMD 