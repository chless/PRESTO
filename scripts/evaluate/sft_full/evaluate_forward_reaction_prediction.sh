#!/bin/bash

# set as environment variables
export HF_HOME="~/.cache/huggingface/"
export MOLECULE_2D_PATH="/home/chanhui-lee/PRESTO/checkpoint/PRESTO"

TASK=forward_reaction_prediction
DATA_DIR="/cto_labs/AIDD/DATA/React/MolInstruct/forward_mmchat_smiles/test"
BASE_LLM_PATH="/home/chanhui-lee/PRESTO/checkpoint/PRESTO"
PROJECTOR_DIR="$BASE_LLM_PATH/non_lora_trainables.bin"

# log path
LOG_DIR="./logs/full/$TASK"

python scripts/evaluate_model.py \
    --model_name_or_path $BASE_LLM_PATH \
    --projectors_path $PROJECTOR_DIR \
    --lora_enable False \
    --dataset_path  $DATA_DIR \
    --max_new_tokens 256 \
    --cache_dir $LOG_DIR \
    --output_dir $LOG_DIR \
    --evaluator "smiles" \
    --verbose \