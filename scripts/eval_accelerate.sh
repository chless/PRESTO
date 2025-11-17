#!/bin/bash
#
# Accelerate-based Multi-GPU Evaluation Script for PRESTO
#
# Usage:
#   bash scripts/eval_accelerate.sh [MODEL_PATH] [DATASET_PATH] [BATCH_SIZE]
#
# Example:
#   bash scripts/eval_accelerate.sh checkpoints/my-model data/test_dataset 8
#

# Default values
MODEL_PATH=${1:-"checkpoints/stage3/llava-moleculestm-vicuna-7b-v1.5-sft"}
DATASET_PATH=${2:-"data/test_dataset"}
BATCH_SIZE=${3:-8}
OUTPUT_DIR=${4:-"results/eval_$(date +%Y%m%d_%H%M%S)"}

# Model configuration
MODEL_CLS="LlamaLMMForCausalLM"
MODALITY_BUILDER="molecule_2d"
EVALUATOR="smiles"
PARSER="base"

# Generation parameters
MAX_NEW_TOKENS=512
TEMPERATURE=0.2
TOP_P=0.95
DO_SAMPLE=false

# Optional paths
LORA_ENABLE=false
PROJECTORS_PATH=""

echo "=================================================="
echo "Accelerate Multi-GPU Evaluation Configuration"
echo "=================================================="
echo "Model Path: $MODEL_PATH"
echo "Dataset Path: $DATASET_PATH"
echo "Batch Size per GPU: $BATCH_SIZE"
echo "Output Directory: $OUTPUT_DIR"
echo "=================================================="
echo ""
echo "Note: Number of GPUs configured in accelerate config"
echo "Run 'accelerate config' if not yet configured"
echo ""
echo "=================================================="

# Create output directory
mkdir -p $OUTPUT_DIR

# Run with accelerate launch
accelerate launch scripts/evaluate_model_accelerate.py \
    --model_name_or_path $MODEL_PATH \
    --model_cls $MODEL_CLS \
    --modality_builder $MODALITY_BUILDER \
    --dataset_path $DATASET_PATH \
    --batch_size $BATCH_SIZE \
    --evaluator $EVALUATOR \
    --parser $PARSER \
    --max_new_tokens $MAX_NEW_TOKENS \
    --temperature $TEMPERATURE \
    --top_p $TOP_P \
    --do_sample $DO_SAMPLE \
    --lora_enable $LORA_ENABLE \
    --output_dir $OUTPUT_DIR \
    --cache_dir $OUTPUT_DIR/cache \
    --verbose false \
    --num_workers 4

echo "Evaluation complete! Results saved to $OUTPUT_DIR"
