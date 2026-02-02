# Training Examples

This directory contains example configuration files for various training scenarios.

## Directory Structure

```
examples/
├── train_full/          # Full fine-tuning examples
├── train_lora/          # LoRA fine-tuning examples
├── train_qlora/         # QLoRA (quantized LoRA) examples
├── merge_lora/          # LoRA weight merging configs
├── inference/           # Inference and chat configs
├── accelerate/          # FSDP and multi-node configs
├── ascend/              # Huawei Ascend NPU configs
├── ktransformers/       # KTransformers optimization configs
├── megatron/            # Megatron-LM training configs
├── v1/                  # V1 engine configs
└── extras/              # Advanced training methods
    ├── adam_mini/       # Adam-mini optimizer
    ├── apollo/          # APOLLO optimizer
    ├── badam/           # BAdam optimizer
    ├── galore/          # GaLore optimizer
    ├── muon/            # Muon optimizer
    ├── fp8/             # FP8 training
    ├── fsdp_qlora/      # FSDP + QLoRA
    ├── loraplus/        # LoRA+ method
    ├── pissa/           # PiSSA method
    ├── oft/             # OFT method
    ├── llama_pro/       # LLaMA Pro
    └── mod/             # Mixture-of-Depths
```

## Usage

Run any example using the CLI:

```bash
llamafactory-cli train examples/train_lora/qwen3_lora_sft.yaml
```

Or use the short command:

```bash
lmf train examples/train_lora/qwen3_lora_sft.yaml
```

## Available Examples

### Basic Training
| Example | Description |
|---------|-------------|
| `train_full/qwen3_full_sft.yaml` | Full fine-tuning |
| `train_lora/qwen3_lora_sft.yaml` | LoRA fine-tuning |
| `train_lora/qwen3_lora_pretrain.yaml` | Continued pre-training with LoRA |
| `train_qlora/qwen3_lora_sft_otfq.yaml` | QLoRA training |

### RLHF Training
| Example | Description |
|---------|-------------|
| `train_lora/qwen3_lora_dpo.yaml` | Direct Preference Optimization |
| `train_lora/qwen3_lora_kto.yaml` | KTO training |
| `train_lora/qwen3_lora_reward.yaml` | Reward model training |

### Multimodal Training
| Example | Description |
|---------|-------------|
| `train_full/qwen3vl_full_sft.yaml` | Vision-language full fine-tuning |
| `train_lora/qwen3vl_lora_sft.yaml` | Vision-language LoRA |
| `train_lora/qwen3vl_lora_dpo.yaml` | Vision-language DPO |

### Model Export
| Example | Description |
|---------|-------------|
| `merge_lora/qwen3_lora_sft.yaml` | Merge LoRA adapters |
| `merge_lora/qwen3_gptq.yaml` | Export with GPTQ quantization |

## Customizing Examples

1. Copy an example config that matches your use case
2. Modify the parameters:
   - `model_name_or_path`: Your base model
   - `dataset`: Your dataset name(s) from dataset_info.json
   - `output_dir`: Where to save the model
   - `template`: Chat template matching your model
   - Training hyperparameters as needed

3. Run your custom config:
```bash
llamafactory-cli train your_config.yaml
```

## Key Configuration Parameters

| Parameter | Description |
|-----------|-------------|
| `model_name_or_path` | Model ID or local path |
| `stage` | Training stage: pt, sft, rm, ppo, dpo, kto |
| `do_train` | Enable training mode |
| `finetuning_type` | full, lora, or freeze |
| `dataset` | Dataset name(s) from dataset_info.json |
| `template` | Chat template (qwen3, llama3, etc.) |
| `cutoff_len` | Maximum sequence length |
| `lora_rank` | LoRA rank (typically 8-64) |
| `lora_target` | Target modules (use "all" for all linear layers) |
| `learning_rate` | Learning rate |
| `num_train_epochs` | Number of training epochs |
| `per_device_train_batch_size` | Batch size per GPU |
| `gradient_accumulation_steps` | Gradient accumulation |
| `bf16` | Use bfloat16 precision |
| `output_dir` | Output directory |
| `logging_steps` | Log every N steps |
| `save_steps` | Save checkpoint every N steps |
