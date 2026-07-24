#!/bin/bash
# Batch evaluation script for open-source VLMs
# Usage: bash run_open_eval.sh qwen      (tmux window 1, GPU 3)
#        bash run_open_eval.sh internvl  (tmux window 2, GPU 0,2)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="${SCRIPT_DIR}/evaluate_safety_benchmark_v2_open.py"
COMMON="--max-samples 450 --request-interval 0 --resume"

run_eval() {
    local gpus=$1 model=$2 views=$3 modality=$4
    echo "=========================================="
    echo "[$(date)] START: model=$model views=$views modality=$modality"
    echo "=========================================="
    CUDA_VISIBLE_DEVICES=$gpus conda run -n habitat python "$SCRIPT" \
        --task all --provider local --local-model "$model" \
        --views $views --modality "$modality" $COMMON
    echo "[$(date)] DONE: model=$model views=$views modality=$modality"
    echo ""
}

if [ "$1" = "qwen" ]; then
    GPU=3

    # --- Qwen2.5-VL-7B (general baseline) ---
    # Stage I: ego views (arm+head), 9 modalities
    run_eval $GPU qwen2.5-vl-7b "arm head" rgb
    run_eval $GPU qwen2.5-vl-7b "arm head" depth
    run_eval $GPU qwen2.5-vl-7b "arm head" depth_color
    run_eval $GPU qwen2.5-vl-7b "arm head" pred_depth
    run_eval $GPU qwen2.5-vl-7b "arm head" pred_depth_color
    run_eval $GPU qwen2.5-vl-7b "arm head" rgbd
    run_eval $GPU qwen2.5-vl-7b "arm head" rgbd_color
    run_eval $GPU qwen2.5-vl-7b "arm head" rgbd_pred
    run_eval $GPU qwen2.5-vl-7b "arm head" rgbd_pred_color

    # Stage II: multi-view configs
    run_eval $GPU qwen2.5-vl-7b "third_robot third_human arm head" rgb
    run_eval $GPU qwen2.5-vl-7b "third_robot third_human arm head" depth_color
    run_eval $GPU qwen2.5-vl-7b "topdown" rgb

    # --- VST-7B-SFT (spatial-specialized, Qwen2.5-VL base) ---
    run_eval $GPU vst-7b-sft "arm head" rgb
    run_eval $GPU vst-7b-sft "arm head" depth
    run_eval $GPU vst-7b-sft "arm head" depth_color
    run_eval $GPU vst-7b-sft "arm head" pred_depth
    run_eval $GPU vst-7b-sft "arm head" pred_depth_color
    run_eval $GPU vst-7b-sft "arm head" rgbd
    run_eval $GPU vst-7b-sft "arm head" rgbd_color
    run_eval $GPU vst-7b-sft "arm head" rgbd_pred
    run_eval $GPU vst-7b-sft "arm head" rgbd_pred_color

    run_eval $GPU vst-7b-sft "third_robot third_human arm head" rgb
    run_eval $GPU vst-7b-sft "third_robot third_human arm head" depth_color
    run_eval $GPU vst-7b-sft "topdown" rgb

elif [ "$1" = "internvl" ]; then
    GPU=0,2

    # --- InternVL3.5-8B (general baseline) ---
    run_eval $GPU internvl3.5-8b "arm head" rgb
    run_eval $GPU internvl3.5-8b "arm head" depth
    run_eval $GPU internvl3.5-8b "arm head" depth_color
    run_eval $GPU internvl3.5-8b "arm head" pred_depth
    run_eval $GPU internvl3.5-8b "arm head" pred_depth_color
    run_eval $GPU internvl3.5-8b "arm head" rgbd
    run_eval $GPU internvl3.5-8b "arm head" rgbd_color
    run_eval $GPU internvl3.5-8b "arm head" rgbd_pred
    run_eval $GPU internvl3.5-8b "arm head" rgbd_pred_color

    run_eval $GPU internvl3.5-8b "third_robot third_human arm head" rgb
    run_eval $GPU internvl3.5-8b "third_robot third_human arm head" depth_color
    run_eval $GPU internvl3.5-8b "topdown" rgb

    # --- SenseNova-SI-1.2-InternVL3-8B (spatial-specialized, InternVL3 base) ---
    run_eval $GPU sensenova-si-1.2-8b "arm head" rgb
    run_eval $GPU sensenova-si-1.2-8b "arm head" depth
    run_eval $GPU sensenova-si-1.2-8b "arm head" depth_color
    run_eval $GPU sensenova-si-1.2-8b "arm head" pred_depth
    run_eval $GPU sensenova-si-1.2-8b "arm head" pred_depth_color
    run_eval $GPU sensenova-si-1.2-8b "arm head" rgbd
    run_eval $GPU sensenova-si-1.2-8b "arm head" rgbd_color
    run_eval $GPU sensenova-si-1.2-8b "arm head" rgbd_pred
    run_eval $GPU sensenova-si-1.2-8b "arm head" rgbd_pred_color

    run_eval $GPU sensenova-si-1.2-8b "third_robot third_human arm head" rgb
    run_eval $GPU sensenova-si-1.2-8b "third_robot third_human arm head" depth_color
    run_eval $GPU sensenova-si-1.2-8b "topdown" rgb

else
    echo "Usage: bash run_open_eval.sh [qwen|internvl]"
    exit 1
fi

echo "=========================================="
echo "[$(date)] ALL DONE"
echo "=========================================="
