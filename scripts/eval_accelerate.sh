# Run with accelerate launch
# Set which GPUs to use (comma-separated list)
export CUDA_VISIBLE_DEVICES=$1
export MOLECULE_2D_PATH="/home/chanhui-lee/PRESTO/checkpoint"
output_dir=$2

accelerate launch scripts/evaluate_model_accelerate.py \
    --model_name_or_path /home/chanhui-lee/PRESTO/checkpoint/PRESTO \
    --projectors_path /home/chanhui-lee/PRESTO/checkpoint/PRESTO/non_lora_trainables.bin \
    --lora_enable False \
    --dataset_path /data/text-mol/data/Mol-LLM-v7.1/mol_llm_testset_presto \
    --batch_size 12 \
    --max_new_tokens 256 \
    --evaluator smiles \
    --output_dir $output_dir

