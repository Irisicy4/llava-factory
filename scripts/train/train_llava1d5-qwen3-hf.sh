#!/bin/bash

set -euo pipefail

# Core model/image configuration
IMAGE_PATH=HF@image
LLM_VERSION=Qwen/Qwen3-8B
VT_VERSION=openai/clip-vit-large-patch14-336
VT_VERSION2=""
CN_VERSION=mlp2x_gelu
CONV_VERSION=qwen2_instruct
VERSION=hf_stream
TRAIN_RECIPE=common
MODEL_MAX_LENGTH=2048
RUN_NAME=hf_stream

# Hugging Face dataset specification
DATA_PATH_SPEC="HF@Icey444/llava_v1_5_mix665k?split=train&streaming=true"
HF_CONVERSATION_COLUMN=conversations
HF_SHUFFLE_BUFFER_SIZE=4096
HF_SHUFFLE_SEED=2025

CUDA_VISIBLE_DEVICES=0,1 \
RUN_NAME=$RUN_NAME \
HF_CONVERSATION_COLUMN=$HF_CONVERSATION_COLUMN \
HF_SHUFFLE_BUFFER_SIZE=$HF_SHUFFLE_BUFFER_SIZE \
HF_SHUFFLE_SEED=$HF_SHUFFLE_SEED \
bash scripts/train/pretrain-hf.sh \
    "$DATA_PATH_SPEC" \
    "$IMAGE_PATH" \
    "$LLM_VERSION" \
    "$VT_VERSION" \
    "$VT_VERSION2" \
    "$CN_VERSION" \
    "$CONV_VERSION" \
    "$VERSION" \
    "$TRAIN_RECIPE" \
    "$MODEL_MAX_LENGTH"
