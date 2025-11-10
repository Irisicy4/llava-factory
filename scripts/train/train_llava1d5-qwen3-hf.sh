DATA_PATH=s3://image/cc_filtered_all3m12m_pretrain_v2.json  #pretrain annotation file path
FINETUNE_DATA_PATH=/tmp/instance_storage/llava/sft/llava_v1_5_mix665k.json   #finetune annotation file path
#IMAGE_PATH=/mnt/zillion/image #pretrain image dir
IMAGE_PATH=s3://image 
FINETUNE_IMAGE_PATH=/tmp/instance_storage/llava/sft/ #finetune image dir

# Streaming dataset configuration (Hugging Face Parquet)
USE_HF_STREAMING=True
HF_DATASET_NAME=Icey444/llava_v1_5_mix665k
HF_DATASET_CONFIG=""
HF_DATASET_SPLIT=train
HF_DATA_FILES=""
HF_CONVERSATION_COLUMN=conversations
HF_IMAGE_COLUMN=image
HF_IMAGE_PATH_COLUMN=image_path
export USE_HF_STREAMING HF_DATASET_NAME HF_DATASET_CONFIG HF_DATASET_SPLIT HF_DATA_FILES HF_CONVERSATION_COLUMN HF_IMAGE_COLUMN HF_IMAGE_PATH_COLUMN

LLM_VERSION=Qwen/Qwen3-8B # llm path in huggingface
VT_VERSION=openai/clip-vit-large-patch14-336 #vision tower path in huggingface
VT_VERSION2="" #if you are not using mof vision tower, keep it empty
CN_VERSION=mlp2x_gelu #connector type, other options are: qformer, resampler, etc
CONV_VERSION=qwen2_instruct #chat template, other options are: phi, llama, gemmma, etc
VERSION=base #experiment name for recording different runnings
TRAIN_RECIPE=common #training recipes, other options are: lora, qlora
MODEL_MAX_LENGTH=2048 #max model length for llm

export CUDA_VISIBLE_DEVICES=0,1
bash scripts/train/pretrain.sh "$DATA_PATH" "$IMAGE_PATH" "$LLM_VERSION" "$VT_VERSION" "$VT_VERSION2" "$CN_VERSION" "$VERSION" "$TRAIN_RECIPE" "$MODEL_MAX_LENGTH"
# bash scripts/train/finetune.sh "$FINETUNE_DATA_PATH" "$FINETUNE_IMAGE_PATH" "$LLM_VERSION" "$VT_VERSION" "$VT_VERSION2" "$CN_VERSION" "$CONV_VERSION" "$VERSION" "$TRAIN_RECIPE" "$MODEL_MAX_LENGTH"
