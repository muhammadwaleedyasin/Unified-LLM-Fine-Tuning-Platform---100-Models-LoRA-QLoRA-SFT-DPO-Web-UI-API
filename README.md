# LLM Fine-Tuning Toolkit

A comprehensive toolkit for fine-tuning large language models with an easy-to-use CLI and Web UI interface.

## Overview

This project provides a unified platform for fine-tuning 100+ large language models including LLaMA, Mistral, Qwen, DeepSeek, Gemma, Phi, and many more. It supports various training methods from full fine-tuning to parameter-efficient techniques like LoRA and QLoRA.

## Features

### Supported Models
- LLaMA (all versions)
- LLaVA (multimodal)
- Mistral / Mixtral-MoE
- Qwen3 / Qwen3-VL
- DeepSeek
- Gemma
- GLM
- Phi
- And 100+ more models

### Training Methods
- **Pre-training**: Continuous pre-training on custom datasets
- **Supervised Fine-Tuning (SFT)**: Standard instruction tuning
- **Multimodal SFT**: Fine-tuning with image/video/audio data
- **Reward Modeling**: Train reward models for RLHF
- **RLHF Methods**: PPO, DPO, KTO, ORPO

### Memory Optimization
- 16-bit full fine-tuning
- Freeze-tuning (partial layer training)
- LoRA (Low-Rank Adaptation)
- QLoRA with 2/3/4/5/6/8-bit quantization
- Support for AQLM, AWQ, GPTQ, LLM.int8, HQQ, EETQ

### Advanced Features
- GaLore optimizer
- BAdam optimizer
- APOLLO optimizer
- Adam-mini
- Muon optimizer
- FlashAttention-2
- Unsloth acceleration
- Liger Kernel
- NEFTune
- LoRA+ and rsLoRA

## Requirements

- Python 3.11+
- PyTorch 2.4.0+
- CUDA 11.8+ (for GPU training)
- 16GB+ RAM recommended
- GPU with 8GB+ VRAM (for LoRA) or 24GB+ (for full fine-tuning)

## Installation

### Using pip

```bash
pip install -e .
```

### Using Docker

```bash
docker pull hiyouga/llamafactory
docker run --gpus all -it hiyouga/llamafactory
```

## Quick Start

### CLI Commands

```
llamafactory-cli api       - Launch an OpenAI-style API server
llamafactory-cli chat      - Launch a chat interface in CLI
llamafactory-cli export    - Merge LoRA adapters and export model
llamafactory-cli train     - Train models
llamafactory-cli webchat   - Launch a chat interface in Web UI
llamafactory-cli webui     - Launch LlamaBoard (full Web UI)
llamafactory-cli env       - Show environment info
llamafactory-cli version   - Show version info
```

You can also use `lmf` as a shortcut for `llamafactory-cli`.

### Training Examples

1. **Supervised Fine-Tuning with LoRA**:
```bash
llamafactory-cli train examples/train_lora/qwen3_lora_sft.yaml
```

2. **Merge LoRA weights**:
```bash
llamafactory-cli export examples/merge_lora/qwen3_lora_sft.yaml
```

3. **Chat with your model**:
```bash
llamafactory-cli chat examples/inference/qwen3_lora_sft.yaml
```

### Web UI (LLaMA Board)

Launch the graphical interface:

```bash
llamafactory-cli webui
```

This opens a Gradio-based web interface where you can:
- Select models and datasets
- Configure training parameters
- Monitor training progress
- Chat with fine-tuned models

## Project Structure

```
.
├── src/llamafactory/     # Main source code
├── data/                 # Dataset configurations and samples
├── examples/             # Example training configs
├── docker/               # Docker configurations
├── scripts/              # Utility scripts
└── tests/                # Test files
```

## Configuration

Training configurations use YAML format. Key parameters:

```yaml
### model
model_name_or_path: Qwen/Qwen3-4B-Instruct-2507
trust_remote_code: true

### method
stage: sft                    # Training stage: sft, pt, rm, ppo, dpo, kto
do_train: true
finetuning_type: lora         # full, lora, or freeze
lora_rank: 8
lora_target: all

### dataset
dataset: alpaca_en_demo
template: qwen3
cutoff_len: 2048
max_samples: 1000

### output
output_dir: saves/model/lora/sft
logging_steps: 10
save_steps: 500
plot_loss: true

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 1.0e-4
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
```

## Multi-GPU Training

For distributed training across multiple GPUs, the toolkit automatically detects and uses all available GPUs. For DeepSpeed:

```bash
llamafactory-cli train examples/train_lora/qwen3_lora_sft_ds3.yaml
```

## API Deployment

Deploy your fine-tuned model with an OpenAI-compatible API:

```bash
llamafactory-cli api examples/inference/qwen3_lora_sft.yaml
```

## Supported Tasks

- Multi-turn dialogue
- Tool/Function calling
- Image understanding
- Visual grounding
- Video recognition
- Audio understanding
- Code generation
- Instruction following

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## Contact

For questions and support, please open an issue in this repository.
