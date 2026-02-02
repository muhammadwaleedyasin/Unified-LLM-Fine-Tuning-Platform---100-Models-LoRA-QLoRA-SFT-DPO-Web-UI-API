# Datasets

This directory contains dataset configurations and sample data for fine-tuning.

## Dataset Configuration

All datasets are registered in `dataset_info.json`. Each entry specifies:
- Dataset name and file path (or Hugging Face/ModelScope dataset ID)
- Data format (alpaca or sharegpt)
- Column mappings
- Optional tags for role/content parsing

## Supported Formats

### Alpaca Format (Default)
Used for instruction-following datasets:
```json
[
  {
    "instruction": "User instruction",
    "input": "Optional context",
    "output": "Expected response"
  }
]
```

### ShareGPT Format
Used for multi-turn conversation datasets:
```json
[
  {
    "conversations": [
      {"from": "human", "value": "User message"},
      {"from": "gpt", "value": "Assistant response"}
    ]
  }
]
```

For multimodal datasets with ShareGPT format:
```json
[
  {
    "messages": [
      {"role": "user", "content": "Describe this image"},
      {"role": "assistant", "content": "The image shows..."}
    ],
    "images": ["path/to/image.jpg"]
  }
]
```

### Preference Format (for DPO/KTO/RLHF)
```json
[
  {
    "instruction": "User instruction",
    "input": "",
    "chosen": "Preferred response",
    "rejected": "Less preferred response"
  }
]
```

## Adding Custom Datasets

1. Place your dataset file in this directory (or use a Hugging Face/ModelScope dataset ID)
2. Add an entry to `dataset_info.json`:

**For local Alpaca-format file:**
```json
{
  "my_dataset": {
    "file_name": "my_data.json"
  }
}
```

**For Hugging Face dataset:**
```json
{
  "my_dataset": {
    "hf_hub_url": "username/dataset_name"
  }
}
```

**For ShareGPT format with custom columns:**
```json
{
  "my_chat_dataset": {
    "file_name": "chat_data.json",
    "formatting": "sharegpt",
    "columns": {
      "messages": "conversations"
    }
  }
}
```

**For multimodal datasets:**
```json
{
  "my_vision_dataset": {
    "file_name": "vision_data.json",
    "formatting": "sharegpt",
    "columns": {
      "messages": "messages",
      "images": "images"
    },
    "tags": {
      "role_tag": "role",
      "content_tag": "content",
      "user_tag": "user",
      "assistant_tag": "assistant"
    }
  }
}
```

3. Use your dataset in training:
```bash
llamafactory-cli train --dataset my_dataset ...
```

Or in a YAML config:
```yaml
dataset: my_dataset
```

## Pre-configured Datasets

The `dataset_info.json` file includes configurations for many popular datasets including:
- `alpaca_en` / `alpaca_zh` - English/Chinese Alpaca datasets
- `alpaca_gpt4_en` / `alpaca_gpt4_zh` - GPT-4 generated Alpaca data
- `identity` - Identity/persona dataset
- `glaive_toolcall_en_demo` - Tool calling examples
- `mllm_demo` - Multimodal (image) examples
- `mllm_video_demo` - Video understanding examples
- `mllm_audio_demo` - Audio understanding examples

Check `dataset_info.json` for the full list of available datasets.
