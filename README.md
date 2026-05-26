# AerialVLA: A Vision-Language-Action Model for Aerial Navigation with Online Dialogue

Aerial Vision-Language-Action model for drone navigation. Based on LLaVA OneVision with Qwen2-7B-Instruct as the language backbone and SigLIP-SO400M as the vision encoder, extended with action prediction heads for waypoint, altitude, and progress estimation in aerial navigation tasks.

## Environment Setup

```bash
# Clone and create environment
git clone https://github.com/chenjinyubuaa/AerialVLA
cd AerialVLA

conda create -n aerialvla python=3.10 -y
conda activate aerialvla

pip install --upgrade pip
pip install -e ".[train]"
```

**Key dependencies**: deepspeed, peft==0.4.0, accelerate, transformers (dev), torch==2.1.2, opencv-python, decord, wandb.

Note: `flash-attn` is optional but recommended for training efficiency. Install via `pip install flash-attn --no-build-isolation`.

## Data Preparation

Download the AVDN dataset and organize as follows:

```
./data/
├── train_data.json                                        # Raw AVDN training annotations
├── train_images/                                          # AVDN training images (*.tif)
├── train_aerial_instrucion_grid_9_altitude_20.json        # Processed instruction data (generated)
├── train_1094.json                                        # Evaluation data
└── val_unseen_data.json                                   # Validation data
```

### Process raw data into instruction format
```bash
python llava/eval/aerialchat/proc_instruction.py
```
This converts raw AVDN annotations into the instruction tuning format with 9×9 grid action space and 20 altitude bins. The output is `train_aerial_instrucion_grid_9_altitude_20.json`.

## Model Weights

Download the pre-trained weights:

```bash
# Base LLaVA OneVision model (from HuggingFace)
# Place at: ./models/llava-onevision-qwen2-7b-ov

# GeoChat stage-1 LoRA checkpoint
# Place at: ./checkpoints/llava-onevision-qwen2-7b-ov-max1-lora-geochat-stage2
```

Expected layout:
```
./models/
└── llava-onevision-qwen2-7b-ov/       # LLaVA OneVision Qwen2-7B base model

./checkpoints/
└── llava-onevision-qwen2-7b-ov-max1-lora-geochat-stage2/   # Stage-1 GeoChat LoRA
```

## Training

Edit `scripts/archived/aerialchat/finetune_lora_remote.sh` to configure experiment name and paths, then run:

```bash
bash scripts/archived/aerialchat/finetune_lora_remote.sh
```

This executes **Stage 2 LoRA fine-tuning** with DAgger self-training:

| Parameter | Value | Description |
|---|---|---|
| `--model_name_or_path` | `./models/llava-onevision-qwen2-7b-ov` | Base model |
| `--data_path` | `./data/train_aerial_instrucion_grid_9_altitude_20.json` | Training data |
| `--image_folder` | `./data/train_images` | Training images |
| `--stage1_path` | `./checkpoints/llava-onevision-qwen2-7b-ov-max1-lora-geochat-stage2` | Stage-1 weights |
| `--lora_enable` | True | Use LoRA |
| `--training_stage` | 2 | Stage 2 (AerialChat) |
| `--use_dagger` | True | Enable DAgger |
| `--dagger_epoch_num` | 20 | DAgger iterations |
| `--dagger_data_rate` | 2.0 | Data augmentation ratio |
| `--use_grid_action` | True | Grid-based actions |
| `--grid_offset_size` | 9 | 9×9 grid for XY offset |
| `--grid_altitude_size` | 20 | Altitude bins |
| `--evaluate_in_training` | True | Eval after each epoch |
| `--num_train_epochs` | 2 | Training epochs |
| `--per_device_train_batch_size` | 4 | Batch size per GPU |
| `--learning_rate` | 2e-5 | Learning rate |
| `--model_max_length` | 32768 | Max sequence length |

Outputs: `./output/aerialchat/{EXP_NAME}/epoch_{N}/`

### Training stages explained

- **Stage 1 (GeoChat)**: General geospatial understanding pre-training. Weights loaded from `./checkpoints/` before Stage 2.
- **Stage 2 (AerialChat DAgger)**: Aerial navigation fine-tuning with iterative self-training. The model generates actions, which are used to augment the training data for the next epoch.

## Evaluation

### Multi-GPU parallel evaluation
```bash
# Edit EXP_NAME in the script first, then run:
bash scripts/archived/aerialchat/eval_batch.sh
```

### Single-GPU evaluation
```bash
python -m llava.eval.aerialchat.inference \
    --model-base ./models/llava-onevision-qwen2-7b-ov \
    --model-path-stage1 ./checkpoints/llava-onevision-qwen2-7b-ov-max1-lora-geochat-stage2 \
    --model-path-stage2 ./output/aerialchat/{EXP_NAME}/epoch_{N} \
    --eval-data-path ./data/train_1094.json \
    --image-folder ./data/train_images \
    --conv-mode qwen_1_5
```

### Evaluation metrics
- **SR** (Success Rate): Overall task success rate
- **TC** (Trajectory Completion): Rate of reaching the destination
- **SPL** (Success weighted by Path Length): Path efficiency
- **IoU Progress**: Intersection over union between current view and target

## DAgger Data Augmentation (standalone)

To augment training data from model outputs without full training:
```bash
python llava/eval/aerialchat/proc_instruction_aug.py
```
Configure paths at the top of the script before running.

## Project Structure

```
.
├── llava/
│   ├── constants.py                     # Token/action constants
│   ├── conversation.py                  # Conversation templates
│   ├── mm_utils.py                      # Multi-modal utilities
│   ├── utils.py                         # General utilities
│   ├── eval/aerialchat/
│   │   ├── inference.py                 # Evaluation entry point
│   │   ├── inference_teacher.py         # Teacher model inference
│   │   ├── proc_instruction.py          # Raw data → instruction format
│   │   ├── proc_instruction_aug.py      # DAgger data augmentation
│   │   └── utils.py                     # GPS/corner/IoU utilities
│   ├── model/
│   │   ├── llava_arch.py                # LLaVA base architecture
│   │   ├── builder.py                   # Model loading
│   │   └── language_model/
│   │       ├── aerialchat_llava_qwen.py  # AerialVLA model (action heads)
│   │       └── llava_qwen.py            # LLaVA-Qwen base
│   └── train/aerialchat/
│       ├── train.py                     # Main training script
│       ├── train_mem.py                 # Training launcher
│       └── aerialchat_trainer.py        # Custom DAgger trainer
├── scripts/archived/aerialchat/
│   ├── finetune_lora_remote.sh          # Main training entry
│   ├── eval_batch.sh                    # Multi-GPU evaluation
│   └── sqa_eval_gather.sh               # Eval result aggregation
├── trl/                                 # Training utilities
└── pyproject.toml                       # Dependencies
```

## Model Architecture

AerialVLA extends LLaVA OneVision with four action prediction heads:

| Head | Dimensions | Description |
|---|---|---|
| Grid Decoder | 81-way (9×9) | XY offset within current view |
| Grid Image Decoder | 81-way (9×9) | Visual-aware grid prediction |
| Altitude Decoder | 20-way | Altitude change classification |
| Progress Decoder | Regression | Distance/IoU progress estimate |

The model takes a history map image and a current view image as input, producing both navigation actions and optional dialogue questions to a human navigator.
