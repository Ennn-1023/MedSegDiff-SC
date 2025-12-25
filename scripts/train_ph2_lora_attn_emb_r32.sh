#!/bin/bash

# PH2 Dataset LoRA Training Script - Attention + Embedding Strategy
# 策略：同時訓練 Attention (QKV + Projection) 和 Embedding 層

# 1. 基本路徑與版本設定
VERSION="old"
DATA_NAME="PH2"
DATA_DIR="./data/PH2/Train"
OUT_DIR="./results/ph2_lora_attn_emb_r32_T50_DPM"
RESUME_CHECKPOINT="./emasavedmodel_step1000.pt"

# 2. LoRA 核心參數
USE_LORA="True"
LORA_RANK=32
LORA_ALPHA=64.0
LORA_DROPOUT=0.1
LORA_TARGET="attn_emb"  # 新增：注入 Attention + Embedding 層

# 3. 訓練超參數
BATCH_SIZE=3
LR=2e-4
SAVE_INTERVAL=500
IMAGE_SIZE=256

# 4. 擴散模型設定
DIFFUSION_STEPS=50
DPM_SOLVER="True"

# 創建輸出目錄
mkdir -p $OUT_DIR

echo "🚀 Starting LoRA Training with Attention + Embedding Strategy"
echo "   Target: $LORA_TARGET (QKV + Projection + Embedding)"
echo "   Rank: $LORA_RANK, Alpha: $LORA_ALPHA"
echo "   Output: $OUT_DIR"

# 執行訓練
python scripts/segmentation_train.py \
    --version $VERSION \
    --data_name $DATA_NAME \
    --data_dir $DATA_DIR \
    --out_dir $OUT_DIR \
    --image_size $IMAGE_SIZE \
    --num_channels 128 \
    --class_cond False \
    --num_res_blocks 2 \
    --num_heads 1 \
    --learn_sigma True \
    --use_scale_shift_norm False \
    --attention_resolutions 16 \
    --diffusion_steps $DIFFUSION_STEPS \
    --dpm_solver $DPM_SOLVER \
    --noise_schedule linear \
    --rescale_learned_sigmas False \
    --rescale_timesteps False \
    --resume_checkpoint $RESUME_CHECKPOINT \
    --use_lora $USE_LORA \
    --lora_rank $LORA_RANK \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --lora_target_modules $LORA_TARGET \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --save_interval $SAVE_INTERVAL \
    --gpu_dev "0"

echo "✅ LoRA Training (Attention + Embedding) completed! Results saved to $OUT_DIR"
