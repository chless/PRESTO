# Quick Start: Multi-GPU Evaluation

Choose your preferred method and get started in 30 seconds!

## Option 1: Accelerate (Recommended for Beginners)

### First Time Setup
```bash
# Install (if needed)
pip install accelerate

# Configure (one-time, interactive)
accelerate config
```

**Answer the prompts:**
- Compute environment: `This machine`
- Machine type: `multi-GPU`
- Number of machines: `1`
- Use DeepSpeed: `no`
- Number of GPUs: `4` (or however many you have)
- Which GPUs: `all`
- Mixed precision: `fp16` (or `bf16` if supported)

### Run Evaluation
```bash
# Simple!
bash scripts/eval_accelerate.sh \
    checkpoints/my-model \
    data/test_dataset \
    8
```

**Or directly:**
```bash
accelerate launch scripts/evaluate_model_accelerate.py \
    --model_name_or_path checkpoints/my-model \
    --dataset_path data/test \
    --batch_size 8
```

---

## Option 2: torchrun (More Control)

### No Setup Needed!

### Run Evaluation
```bash
# Full control
bash scripts/eval_multigpu.sh \
    4 \  # number of GPUs
    checkpoints/my-model \
    data/test_dataset \
    8  # batch size
```

**Or directly:**
```bash
torchrun --nproc_per_node=4 scripts/evaluate_model_multigpu.py \
    --model_name_or_path checkpoints/my-model \
    --dataset_path data/test \
    --batch_size 8
```

---

## Common Use Cases

### Use Only 2 GPUs
```bash
# Accelerate: edit config or
CUDA_VISIBLE_DEVICES=0,1 accelerate launch scripts/evaluate_model_accelerate.py ...

# torchrun:
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 scripts/evaluate_model_multigpu.py ...
```

### Use Specific GPUs (e.g., 4,5,6,7)
```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 bash scripts/eval_multigpu.sh 4 ...
# or
CUDA_VISIBLE_DEVICES=4,5,6,7 bash scripts/eval_accelerate.sh ...
```

### Greedy Decoding (Reproducible)
```bash
accelerate launch scripts/evaluate_model_accelerate.py \
    --do_sample false \
    --temperature 0.0 \
    ...
```

### Sampling (Diverse Outputs)
```bash
accelerate launch scripts/evaluate_model_accelerate.py \
    --do_sample true \
    --temperature 0.8 \
    --top_p 0.95 \
    ...
```

---

## Troubleshooting

### "CUDA out of memory"
```bash
# Reduce batch size
--batch_size 4  # or even 2 or 1
```

### "Address already in use" (torchrun)
```bash
torchrun --nproc_per_node=4 --master_port=29501 ...
```

### Accelerate not configured
```bash
accelerate config  # Run this first!
```

### Check results match
```bash
python scripts/compare_eval_results.py \
    --result1 results/run1/cache \
    --result2 results/run2/cache
```

---

## Files Created

You now have:
- ✅ `scripts/evaluate_model_multigpu.py` - torchrun version
- ✅ `scripts/evaluate_model_accelerate.py` - accelerate version
- ✅ `scripts/eval_multigpu.sh` - torchrun wrapper script
- ✅ `scripts/eval_accelerate.sh` - accelerate wrapper script
- ✅ `scripts/compare_eval_results.py` - result comparison tool
- ✅ `MULTIGPU_EVAL_README.md` - Full documentation
- ✅ `TORCHRUN_VS_ACCELERATE.md` - Detailed comparison

---

## Quick Decision Tree

```
Do you need multi-GPU evaluation?
│
├─ Yes ──> New to distributed computing?
│         │
│         ├─ Yes ──> Use accelerate ✅
│         │          (cleaner, easier)
│         │
│         └─ No ──> Want maximum control?
│                  │
│                  ├─ Yes ──> Use torchrun ✅
│                  │          (full control)
│                  │
│                  └─ No ──> Use accelerate ✅
│                            (less boilerplate)
│
└─ No ──> Use scripts/evaluate_model.py
          (original single-GPU version)
```

---

## Performance Expectations

**Example: 1000 samples, 7B model, A100 GPUs**

| Setup | Time | Speedup |
|-------|------|---------|
| 1 GPU, batch=1 | ~60 min | 1x |
| 1 GPU, batch=8 | ~15 min | 4x |
| 4 GPUs, batch=8 (accelerate) | ~4 min | 15x |
| 4 GPUs, batch=8 (torchrun) | ~4 min | 15x |
| 8 GPUs, batch=8 | ~2 min | 30x |

**Both accelerate and torchrun have identical performance!**

---

## Next Steps

1. **Test on small dataset first:**
   ```bash
   # Create small test
   head -n 100 data/full_dataset > data/small_test

   # Run quick test
   bash scripts/eval_accelerate.sh checkpoints/model data/small_test 4
   ```

2. **Compare with single-GPU baseline:**
   ```bash
   python scripts/compare_eval_results.py \
       --result1 results/single_gpu/cache \
       --result2 results/multi_gpu/cache
   ```

3. **Scale up to full dataset:**
   ```bash
   bash scripts/eval_accelerate.sh checkpoints/model data/full_dataset 8
   ```

---

**Happy evaluating! 🚀**

For detailed documentation, see:
- `MULTIGPU_EVAL_README.md` - Complete guide
- `TORCHRUN_VS_ACCELERATE.md` - Detailed comparison
