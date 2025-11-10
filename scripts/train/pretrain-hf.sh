#!/bin/bash

set -euo pipefail

if [ $# -ne 10 ]; then
    echo "Usage: $0 <DATA_PATH> <IMAGE_PATH> <LLM_VERSION> <VT_VERSION> <VT_VERSION2> <CN_VERSION> <CONV_VERSION> <VERSION> <TRAIN_RECIPE> <MODEL_MAX_LENGTH>"
    exit 1
fi

DATA_PATH_INPUT="$1"
RAW_IMAGE_PATH="$2"
IMAGE_PATH="$RAW_IMAGE_PATH"
LLM_VERSION="$3"
VT_VERSION="$4"
VT_VERSION2="$5"
CN_VERSION="$6"
CONV_VERSION="$7"
VERSION="$8"
TRAIN_RECIPE="$9"
MODEL_MAX_LENGTH="${10}"

VT_VARIANT="${VT_VERSION#*/}"
LLM_VARIANT="${LLM_VERSION#*/}"

USE_HF_DATASET=${USE_HF_DATASET:-False}
HF_DATASET_NAME=${HF_DATASET_NAME:-""}
HF_DATASET_CONFIG=${HF_DATASET_CONFIG:-""}
HF_DATASET_SPLIT=${HF_DATASET_SPLIT:-train}
HF_DATA_FILES=${HF_DATA_FILES:-""}
HF_STREAMING=${HF_STREAMING:-True}
HF_CACHE_DIR=${HF_CACHE_DIR:-""}
HF_CONVERSATION_COLUMN=${HF_CONVERSATION_COLUMN:-conversations}
HF_IMAGE_COLUMN=${HF_IMAGE_COLUMN:-}
HF_SHUFFLE_BUFFER_SIZE=${HF_SHUFFLE_BUFFER_SIZE:-4096}
HF_SHUFFLE_SEED=${HF_SHUFFLE_SEED:-42}
RUN_NAME=${RUN_NAME:-hf_stream}

DATA_PATH_ARG=()
if [ -n "$DATA_PATH_INPUT" ]; then
    if [[ "$DATA_PATH_INPUT" == HF@* ]]; then
        USE_HF_DATASET=True
        hf_spec="${DATA_PATH_INPUT#HF@}"
        dataset_id="$hf_spec"
        query=""
        if [[ "$hf_spec" == *\?* ]]; then
            dataset_id="${hf_spec%%\?*}"
            query="${hf_spec#*\?}"
        fi
        HF_DATASET_NAME="$dataset_id"
        if [ -n "$query" ]; then
            IFS='&' read -ra params <<< "$query"
            for param in "${params[@]}"; do
                [[ -z "$param" ]] && continue
                key="${param%%=*}"
                value="${param#*=}"
                case "$key" in
                    split) HF_DATASET_SPLIT="$value" ;;
                    config) HF_DATASET_CONFIG="$value" ;;
                    files) HF_DATA_FILES="$value" ;;
                    streaming) HF_STREAMING="$value" ;;
                    cache_dir) HF_CACHE_DIR="$value" ;;
                esac
            done
        fi
    else
        DATA_PATH_ARG=(--data_path "$DATA_PATH_INPUT")
    fi
elif [ -n "${DATA_PATH:-}" ]; then
    DATA_PATH_ARG=(--data_path "$DATA_PATH")
fi

if [[ "$RAW_IMAGE_PATH" == HF@* ]]; then
    HF_IMAGE_COLUMN="${RAW_IMAGE_PATH#HF@}"
    IMAGE_PATH="."
fi

HF_IMAGE_COLUMN=${HF_IMAGE_COLUMN:-image}

deepspeed --master_port 29501 tinyllava/train/train.py \
    --deepspeed ./scripts/zero3.json \
    "${DATA_PATH_ARG[@]}" \
    --image_folder $IMAGE_PATH \
    --is_multimodal True \
    --conv_version $CONV_VERSION \
    --model_name_or_path $LLM_VERSION \
    --vision_tower $VT_VERSION \
    --vision_tower2 "$VT_VERSION2" \
    --connector_type $CN_VERSION \
    --mm_vision_select_layer -2 \
    --image_aspect_ratio square \
    --attn_implementation flash_attention_2 \
    --fp16 False \
    --bf16 True \
    --training_recipe $TRAIN_RECIPE \
    --tune_type_llm frozen \
    --tune_type_vision_tower frozen \
    --tune_vision_tower_from_layer 0 \
    --tune_type_connector full \
    --output_dir ./work_dirs/llava-${LLM_VARIANT}-${VT_VARIANT}-${VERSION} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length $MODEL_MAX_LENGTH \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --tokenizer_use_fast False \
    --run_name $RUN_NAME \
    --use_hf_dataset $USE_HF_DATASET \
    --hf_dataset_name "$HF_DATASET_NAME" \
    --hf_dataset_config "$HF_DATASET_CONFIG" \
    --hf_dataset_split $HF_DATASET_SPLIT \
    --hf_data_files "$HF_DATA_FILES" \
    --hf_streaming $HF_STREAMING \
    --hf_conversation_column $HF_CONVERSATION_COLUMN \
    --hf_image_column $HF_IMAGE_COLUMN \
    --hf_cache_dir "$HF_CACHE_DIR" \
    --hf_shuffle_buffer_size $HF_SHUFFLE_BUFFER_SIZE \
    --hf_shuffle_seed $HF_SHUFFLE_SEED
