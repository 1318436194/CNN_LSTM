#!/bin/bash

# CNN-LSTM模型的超参数调优

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
    --output_dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --n_trials)
      N_TRIALS="$2"
      shift 2
      ;;
    *)
      # 未知选项
      echo "未知选项: $1"
      exit 1
      ;;
  esac
done

# 构建命令
CMD="python main.py --mode tune --config ${CONFIG_FILE}"

# 如果提供了可选参数，则添加
if [ ! -z "$DATA_PATH" ]; then
  CMD="$CMD --data_path $DATA_PATH"
fi

if [ ! -z "$OUTPUT_DIR" ]; then
  CMD="$CMD --output_dir $OUTPUT_DIR"
fi

if [ ! -z "$N_TRIALS" ]; then
  CMD="$CMD --n_trials $N_TRIALS"
fi

# 打印命令
echo "执行: $CMD"

# 执行命令
eval $CMD 