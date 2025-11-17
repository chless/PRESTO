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
    output_dir: str = field(default='./evaluation_dump', metadata={"help": "Path to the output file."})
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
        shuffle=True,
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

    if accelerator.is_main_process:
        pbar = tqdm.tqdm(total=len(dataloader), desc="Evaluating")

    for batch in dataloader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        modality_inputs = batch['modality_inputs']

        with torch.inference_mode():
            # Unwrap model for generation (DDP doesn't support generate)
            unwrapped_model = accelerator.unwrap_model(model)

            output_dict = unwrapped_model.generate(
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
                output_logits=True,
            )

        output_ids = output_dict.sequences

        batch_size, _ = output_dict.sequences.shape
        sequence_length = len(output_dict.logits)
        vocab_size = output_dict.logits[0].shape[-1]
        # stack logtis
        logits_stacked = torch.zeros(
            batch_size,
            0,
            vocab_size,
            device=output_dict.logits[0].device,
        )
        for i in range(sequence_length):
            logits = output_dict.logits[i].unsqueeze(1)
            logits = (
                logits.view(batch_size, 1, -1).max(dim=1).values.unsqueeze(1)
            )
            logits_stacked = torch.cat([logits_stacked, logits], dim=1)

        tasks = batch['task']
        input_texts = [tokenizer.decode(torch.where(ids>0, ids, tokenizer.pad_token_id), skip_special_tokens=True) for ids in input_ids]
        targets = batch['ground_truths']
        predictions = []
        binary_classificaiton_probs = convert_logit2binary_prob(logits_stacked, tokenizer, tasks)

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

            predictions.append(generated_text)

        save_dict = {
            'tasks': tasks,
            'input_texts': input_texts,
            'targets': targets,
            'predictions': predictions,
            'binary_classificaiton_probs': binary_classificaiton_probs,
            "output_dir": args.output_dir,
        }
        output_dir = args.output_dir or "evaluation_outputs"
        # save tokenizer
        tokenizer_path = os.path.join(output_dir, "tokenizer")
        tokenizer.save_pretrained(tokenizer_path)

        save_predictions(**save_dict)

        if accelerator.is_main_process:
            pbar.update(1)

    if accelerator.is_main_process:
        pbar.close()


def save_predictions(**kwargs):
    output_dir = kwargs.pop('output_dir', 'evaluation_dump')
    os.makedirs(output_dir, exist_ok=True)
    # get rank
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    keys = list(kwargs.keys())
    len_dump = len(kwargs[keys[0]])
    for k in keys:
        assert len(kwargs[k]) == len_dump

    filepath = os.path.join(output_dir, f'dumps_rank_{rank}.json')
    # load the previous dumps from filepath
    if os.path.exists(filepath):
        # read jsonl file
        with open(filepath, 'r', encoding='utf8') as f:
            cumulative_dumps = json.load(f)
    else:
        cumulative_dumps = []

    for i in range(len_dump):
        line = {k: kwargs[k][i] for k in keys}
        cumulative_dumps.append(line)

    with open(filepath, 'w', encoding='utf8') as f:
        # save json file
        json.dump(cumulative_dumps, f, ensure_ascii=True, indent=4)

def convert_logit2binary_prob(logits, tokenizer, tasks):
    """Convert model logits into binary classification probabilities (True / False)."""

    classification_classes = {
        'bace',
        'smol-property_prediction-bbbp',
        'smol-property_prediction-clintox',
        'smol-property_prediction-hiv',
        'smol-property_prediction-sider',
    }

    # Create mask for which tasks are binary classification tasks
    classification_masks = [any(cls in task for cls in classification_classes) for task in tasks]
    classification_masks = torch.tensor(classification_masks, dtype=torch.bool, device=logits.device).unsqueeze(1)

    # Prepare token IDs (move them to same device as logits)
    positive_tokens = ["True", "true", "TRUE", "yes", "Yes", "YES"]
    negative_tokens = ["False", "false", "FALSE", "no", "No", "NO"]

    positive_token_ids = [tokenizer.encode(tok)[1] for tok in positive_tokens]
    negative_token_ids = [tokenizer.encode(tok)[1] for tok in negative_tokens]

    # Convert to tensors on same device
    positive_token_ids = torch.tensor(positive_token_ids, dtype=torch.long, device=logits.device)
    negative_token_ids = torch.tensor(negative_token_ids, dtype=torch.long, device=logits.device)

    # Compute probabilities
    probs = logits.softmax(dim=-1)
    batch_size, seq_len, _ = probs.size()

    false_logits = torch.zeros(batch_size, 1, device=logits.device)
    true_logits = torch.zeros(batch_size, 1, device=logits.device)
    target_logits_index = torch.zeros(batch_size, dtype=torch.long, device=logits.device)

    for i in range(batch_size):
        logits_i = logits[i]
        prediction_ids_i = logits_i.argmax(dim=-1)  # shape: [seq_len]

        # Find indices of any tokens matching positive/negative token IDs
        mask_match = torch.isin(prediction_ids_i, torch.cat((positive_token_ids, negative_token_ids)))

        if mask_match.any():
            first_match_idx = torch.nonzero(mask_match, as_tuple=False)[0, 0]
            target_logits_index[i] = first_match_idx
        else:
            target_logits_index[i] = 0

        # Use .sum() properly across the vocabulary dim
        false_logits[i] = probs[i, target_logits_index[i], negative_token_ids].sum()
        true_logits[i]  = probs[i, target_logits_index[i], positive_token_ids].sum()

    # Stack probabilities and normalize
    total_probs = torch.cat([false_logits, true_logits], dim=-1)
    total_probs = total_probs.softmax(dim=-1)

    # Fill non-classification tasks with -1
    total_probs = torch.where(classification_masks, total_probs, torch.full_like(total_probs, -1.0))

    # Convert to list for output
    return total_probs.tolist()

if __name__ == "__main__":
    main()
