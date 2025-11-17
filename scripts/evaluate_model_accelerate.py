"""
Multi-GPU Evaluation using HuggingFace Accelerate (Simpler Alternative)

Usage:
    # First time: configure
    accelerate config

    # Then run
    accelerate launch scripts/evaluate_model_accelerate.py \
        --model_name_or_path checkpoints/my-model \
        --dataset_path data/test \
        --batch_size 8
"""

from dataclasses import dataclass, field
import logging
import json
import os
from typing import Dict, List

from datasets import load_from_disk, load_dataset, Dataset as HFDataset
import transformers
import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import gather_object
import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from presto.training import ModelArguments
from presto.inference import load_trained_lora_model, load_trained_model
from presto.data_tools import encode_chat, parse_chat_output, encode_interleaved_data
from presto.chemistry_tools import EVALUATOR_BUILDERS


@dataclass
class EvaluationArguments(ModelArguments):
    dataset_path: str = field(default=None, metadata={"help": "Path to evaluation data."})
    lora_enable: bool = field(default=True, metadata={"help": "Enable LoRA."})
    max_new_tokens: int = field(default=2048, metadata={"help": "Maximum number of new tokens to generate."})
    temperature: float = field(default=0.2, metadata={"help": "Temperature to use for sampling."})
    top_k: int = field(default=50, metadata={"help": "Top k to use for sampling."})
    top_p: float = field(default=0.8, metadata={"help": "Top p to use for sampling."})
    do_sample: bool = field(default=True, metadata={"help": "Whether to sample from the output distribution."})
    load_bits: int = field(default=16, metadata={"help": "Quantization bits to use."})
    parser: str = field(default='base', metadata={"help": "Parser for the generated output."})
    evaluator: str = field(default='smiles', metadata={"help": "Evaluator to use for the generated output."})
    cache_dir: str = field(default=None, metadata={"help": "Path to the cache directory."})
    output_dir: str = field(default=None, metadata={"help": "Path to the output file."})
    is_icl: bool = field(default=False, metadata={"help": "Whether ICL testing is enabled."})
    verbose: bool = field(default=False, metadata={"help": "Print verbose output."})
    batch_size: int = field(default=1, metadata={"help": "Batch size per GPU."})
    num_workers: int = field(default=4, metadata={"help": "DataLoader workers."})


class EvaluationDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, tokenizer, modalities, chat_template, is_icl=False):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.modalities = modalities
        self.chat_template = chat_template
        self.is_icl = is_icl

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        entry = self.dataset[idx]

        if self.is_icl:
            encoded_dict = encode_interleaved_data(entry, self.tokenizer, self.modalities)
        else:
            encoded_dict = encode_chat(entry, self.tokenizer, self.modalities, self.chat_template)

        encoded_dict['ground_truth'] = entry['ground_truth']
        encoded_dict['idx'] = idx
        
        encoded_dict['task'] = entry.get('task', 'unknown')
        return encoded_dict


def collate_fn(batch, modalities, tokenizer, pad_token_id=0):
    max_len = max(item['input_ids'].shape[0] for item in batch)
    batch_size = len(batch)

    input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)

    modality_inputs = {m.name: [] for m in modalities}
    ground_truths = []
    indices = []

    for i, item in enumerate(batch):
        seq_len = item['input_ids'].shape[0]
        #input_ids[i, :seq_len] = item['input_ids']
        #attention_mask[i, :seq_len] = 1
        for m in modalities:
            modality_inputs[m.name].append(item[m.name])

        ground_truths.append(item['ground_truth'])
        indices.append(item['idx'])

    tokenizer.padding_side = "left"
    input_ids_list = [item['input_ids'] for item in batch]
    padded = tokenizer.pad(
        {'input_ids': input_ids_list},
        padding=True,
        return_tensors='pt',
        return_attention_mask=True
    )
    input_ids = padded['input_ids']
    attention_mask = padded['attention_mask']

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'modality_inputs': modality_inputs,
        'ground_truths': ground_truths,
        'indices': indices,
        'seq_lens': [item['input_ids'].shape[0] for item in batch],
        'task': [item.get('task', 'unknown') for item in batch],
    }


def main():
    # Parse arguments
    parser = transformers.HfArgumentParser((EvaluationArguments,))
    args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    # Initialize Accelerator - THIS IS THE KEY DIFFERENCE!
    # Handles all distributed setup automatically
    accelerator = Accelerator()

    # Setup logging (only on main process)
    if accelerator.is_main_process:
        logging.basicConfig(level=logging.INFO)
        logging.info(f"Starting evaluation on {accelerator.num_processes} GPUs")
        logging.info(f"Batch size per GPU: {args.batch_size}")
        logging.info(f"Effective batch size: {args.batch_size * accelerator.num_processes}")

    # Load dataset
    if os.path.exists(args.dataset_path):
        try:
            dataset = load_from_disk(args.dataset_path)
        except:
            dataset = load_dataset(args.dataset_path, split="test")
    else:
        dataset = load_dataset(args.dataset_path, split="test")

    if accelerator.is_main_process:
        logging.info(f"Dataset loaded: {len(dataset)} samples")

    # Load model
    if args.lora_enable:
        model, tokenizer = load_trained_lora_model(
            model_name_or_path=args.model_name_or_path,
            model_lora_path=args.model_lora_path,
            load_bits=args.load_bits,
        )
    else:
        model, tokenizer = load_trained_model(
            model_name_or_path=args.model_name_or_path,
            pretrained_projectors_path=args.projectors_path,
            load_bits=args.load_bits,
        )

    model.eval()

    # Setup dataset
    llama2_chat_template = transformers.AutoTokenizer.from_pretrained(
        "meta-llama/Llama-2-7b-chat-hf"
    ).get_chat_template()

    eval_dataset = EvaluationDataset(
        dataset, tokenizer, model.modalities, llama2_chat_template, args.is_icl
    )

    dataloader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda batch: collate_fn(batch, model.modalities, tokenizer, tokenizer.pad_token_id or tokenizer.eos_token_id),
        pin_memory=True
    )

    # Prepare with accelerator - handles DDP automatically!
    model, dataloader = accelerator.prepare(model, dataloader)

    # Move modalities to the same device as the model
    # Get the actual model (unwrap if it's wrapped by DDP)
    unwrapped_model = accelerator.unwrap_model(model)
    model_device = next(unwrapped_model.parameters()).device

    for modality in unwrapped_model.modalities:
        modality.to(device=model_device, dtype=modality.dtype)

    if accelerator.is_main_process:
        logging.info(f"Moved modalities to device: {model_device}")

    # Collect predictions
    all_predictions = []
    all_references = []
    all_indices = []

    if accelerator.is_main_process:
        pbar = tqdm.tqdm(total=len(dataloader), desc="Evaluating")

    for batch in dataloader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        modality_inputs = batch['modality_inputs']
        tasks = batch['task']

        with torch.inference_mode():
            # Unwrap model for generation (DDP doesn't support generate)
            unwrapped_model = accelerator.unwrap_model(model)

            output = unwrapped_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                top_k=args.top_k,
                top_p=args.top_p,
                do_sample=args.do_sample,
                temperature=args.temperature,
                modality_inputs=modality_inputs,
                return_dict_in_generate=True,
                output_scores=True,
            )

            output_ids = output.sequences

        # Decode
        for i, (output, seq_len) in enumerate(zip(output_ids, batch['seq_lens'])):
            prompt_seq_len = input_ids[i].shape[0]
            generated_tokens = output[prompt_seq_len:]
            generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

            if args.parser:
                try:
                    generated_text = parse_chat_output(generated_text, args.parser)["output"]
                except:
                    pass

            all_predictions.append(generated_text)
            all_references.append(batch['ground_truths'][i])
            all_indices.append(batch['indices'][i])

        if accelerator.is_main_process:
            pbar.update(1)

    if accelerator.is_main_process:
        pbar.close()

    # Gather results from all GPUs - accelerate makes this easy!
    all_predictions = gather_object(all_predictions)
    all_references = gather_object(all_references)
    all_indices = gather_object(all_indices)

    # Only main process computes metrics
    if accelerator.is_main_process:
        # Sort by original index
        sorted_results = sorted(zip(all_indices, all_predictions, all_references))
        _, final_predictions, final_references = zip(*sorted_results)

        final_predictions = list(final_predictions)
        final_references = list(final_references)

        # Save results
        if args.cache_dir:
            os.makedirs(args.cache_dir, exist_ok=True)
            with open(os.path.join(args.cache_dir, "predictions.txt"), "w") as f:
                f.write("\n".join(final_predictions))
            with open(os.path.join(args.cache_dir, "references.txt"), "w") as f:
                f.write("\n".join(final_references))
            with open(os.path.join(args.cache_dir, "results.json"), "w") as f:
                json.dump({"predictions": final_predictions, "references": final_references}, f, indent=2)

        # Evaluate
        evaluator = EVALUATOR_BUILDERS[args.evaluator]()
        score = evaluator.evaluate(final_predictions, final_references, verbose=True)

        logging.info(f"Results: {score}")

        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            with open(os.path.join(args.output_dir, "score.json"), "w") as f:
                json.dump(score, f, indent=2)


if __name__ == "__main__":
    main()
