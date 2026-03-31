#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

DATA_NAME="${DATA_NAME:-aime}"
SPLIT="${SPLIT:-test}"
START_IDX="${START_IDX:-0}"
END_IDX="${END_IDX:--1}"
MAX_TOKENS="${MAX_TOKENS:-32768}"
WAIT_FOLLOWUP_MAX_TOKENS="${WAIT_FOLLOWUP_MAX_TOKENS:-$MAX_TOKENS}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_DIR/eval/outputs_section_3_4_3}"

DEEPSEEK_1_5B_MODEL="${DEEPSEEK_1_5B_MODEL:-deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}"

# Paper-aligned decoding choice for the DeepSeek Distill family:
# - temperature 0.7, top_p 0.8

echo "Repository: $REPO_DIR"
echo "Outputs:    $OUTPUT_DIR"
echo "Dataset:    $DATA_NAME/$SPLIT [$START_IDX:$END_IDX]"
echo "Model:      $DEEPSEEK_1_5B_MODEL"
echo ""

run_eval() {
  echo "Running: $*"
  "$PYTHON_BIN" "$REPO_DIR/eval/eval_section_3_4_3.py" "$@"
  echo ""
}

run_deepseek_controls() {
  local model="$1"

  run_eval \
    --model_name_or_path "$model" \
    --control_mode baseline \
    --data_name "$DATA_NAME" \
    --split "$SPLIT" \
    --start_idx "$START_IDX" \
    --end_idx "$END_IDX" \
    --temperature 0.7 \
    --top_p 0.8 \
    --max_tokens "$MAX_TOKENS" \
    --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" \
    --max_model_len "$MAX_MODEL_LEN" \
    --prompt_type qwen-instruct \
    --output_dir "$OUTPUT_DIR" \
    --surround_with_messages

  run_eval \
    --model_name_or_path "$model" \
    --control_mode induce \
    --induction_shots 1 \
    --data_name "$DATA_NAME" \
    --split "$SPLIT" \
    --start_idx "$START_IDX" \
    --end_idx "$END_IDX" \
    --temperature 0.7 \
    --top_p 0.8 \
    --max_tokens "$MAX_TOKENS" \
    --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" \
    --max_model_len "$MAX_MODEL_LEN" \
    --prompt_type qwen-instruct \
    --output_dir "$OUTPUT_DIR" \
    --surround_with_messages

  run_eval \
    --model_name_or_path "$model" \
    --control_mode induce \
    --induction_shots 2 \
    --data_name "$DATA_NAME" \
    --split "$SPLIT" \
    --start_idx "$START_IDX" \
    --end_idx "$END_IDX" \
    --temperature 0.7 \
    --top_p 0.8 \
    --max_tokens "$MAX_TOKENS" \
    --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" \
    --max_model_len "$MAX_MODEL_LEN" \
    --prompt_type qwen-instruct \
    --output_dir "$OUTPUT_DIR" \
    --surround_with_messages

  run_eval \
    --model_name_or_path "$model" \
    --control_mode insert_wait \
    --data_name "$DATA_NAME" \
    --split "$SPLIT" \
    --start_idx "$START_IDX" \
    --end_idx "$END_IDX" \
    --temperature 0.7 \
    --top_p 0.8 \
    --max_tokens "$MAX_TOKENS" \
    --wait_followup_max_tokens "$WAIT_FOLLOWUP_MAX_TOKENS" \
    --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" \
    --max_model_len "$MAX_MODEL_LEN" \
    --prompt_type qwen-instruct \
    --output_dir "$OUTPUT_DIR" \
    --surround_with_messages

  run_eval \
    --model_name_or_path "$model" \
    --control_mode suppress \
    --data_name "$DATA_NAME" \
    --split "$SPLIT" \
    --start_idx "$START_IDX" \
    --end_idx "$END_IDX" \
    --temperature 0.7 \
    --top_p 0.8 \
    --max_tokens "$MAX_TOKENS" \
    --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" \
    --max_model_len "$MAX_MODEL_LEN" \
    --prompt_type qwen-instruct \
    --output_dir "$OUTPUT_DIR" \
    --surround_with_messages
}

# Run only the lightweight DeepSeek model by default.
run_deepseek_controls "$DEEPSEEK_1_5B_MODEL"
