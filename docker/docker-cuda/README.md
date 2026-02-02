# Docker CUDA Setup

Docker configuration for running the toolkit with NVIDIA GPU support.

## Prerequisites

- Docker installed
- NVIDIA Container Toolkit installed
- NVIDIA GPU with CUDA support

## Building the Image

```bash
docker build -t llm-finetune:cuda -f Dockerfile .
```

## Running the Container

### Interactive Mode
```bash
docker run --gpus all -it llm-finetune:cuda
```

### With Volume Mount (for data persistence)
```bash
docker run --gpus all -it \
    -v /path/to/data:/app/data \
    -v /path/to/output:/app/output \
    llm-finetune:cuda
```

### With Web UI
```bash
docker run --gpus all -it \
    -p 7860:7860 \
    llm-finetune:cuda \
    llamafactory-cli webui --server_name 0.0.0.0
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `CUDA_VISIBLE_DEVICES` | GPU devices to use | all |
| `HF_HOME` | Hugging Face cache directory | /root/.cache/huggingface |

## Resource Requirements

- Minimum 16GB GPU VRAM for LoRA training
- Minimum 24GB GPU VRAM for full fine-tuning
- Recommended 32GB+ system RAM
