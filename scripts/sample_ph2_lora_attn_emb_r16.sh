#!/bin/bash

# PH2 Dataset LoRA Sampling Script - Attention + Embedding Strategy
# 對應訓練腳本：train_ph2_lora_attn_emb.sh

# 設定模型路徑（請根據實際訓練結果修改）
MODEL_PATH="./results/ph2_lora_attn_emb_r16_T50_DPM/model001000.pt"
DATA_DIR="./data/PH2/Test"
OUT_DIR="./results/predictions_PH2_lora_attn_emb_r16_T50_DPM"

# LoRA 設定（必須與訓練時一致）
LORA_TARGET="attn_emb"  # Attention + Embedding
LORA_RANK=16
LORA_ALPHA=32.0

# 擴散模型設定
DIFFUSION_STEPS=50
DPM_SOLVER="True"
NUM_ENSEMBLE=5

# 創建輸出目錄
mkdir -p $OUT_DIR

echo "🔮 Starting LoRA Sampling with Attention + Embedding Strategy"
echo "   Model: $MODEL_PATH"
echo "   Target: $LORA_TARGET"
echo "   Output: $OUT_DIR"

# 執行採樣
python scripts/segmentation_sample.py \
    --version old \
    --data_name PH2 \
    --data_dir $DATA_DIR \
    --out_dir $OUT_DIR \
    --model_path $MODEL_PATH \
    --image_size 256 \
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
    --num_ensemble $NUM_ENSEMBLE \
    --use_lora True \
    --lora_rank $LORA_RANK \
    --lora_alpha $LORA_ALPHA \
    --lora_target_modules $LORA_TARGET \
    --gpu_dev "0"

echo "✅ Sampling completed! Predictions saved to $OUT_DIR"
