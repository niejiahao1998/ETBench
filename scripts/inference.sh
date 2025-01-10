#!/bin/bash

set -e

stage3_path=${1:-"./save_model/etchat-stage-3-mlp-dc"}
# stage3_path=${1:-"./save_model/etchat-stage-3-mlp-zgj"}

export CUDA_VISIBLE_DEVICES=0,1
export PYTHONPATH="./:$PYTHONPATH"

IFS="," read -ra GPULIST <<< "${CUDA_VISIBLE_DEVICES:-0}"
CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python etchat/eval/infer_etbench.py \
        --anno_path /home/han023/project/ETBench/Modified/huggingface/PolyU-ChenLab/ETBench/annotations/vid \
        --data_path /home/han023/project/ETBench/Modified/huggingface/PolyU-ChenLab/ETBench/videos_compressed \
        --pred_path $stage3_path/etbench \
        --model_path $stage3_path \
        --chunk $CHUNKS \
        --index $IDX \
        --verbose &
done

wait
