#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

OUTPUT_DIR="${OUTPUT_DIR:-$REPO_DIR/eval/outputs_section_3_4_3}"
PLOT_OUTPUT="${PLOT_OUTPUT:-$REPO_DIR/eval/plots/figure6_section_3_4_3.png}"
DATA_NAME="${DATA_NAME:-}"
SPLIT="${SPLIT:-}"
START_IDX="${START_IDX:-}"
END_IDX="${END_IDX:-}"

ARGS=(
  --output_dir "$OUTPUT_DIR"
  --output "$PLOT_OUTPUT"
)

if [[ -n "$DATA_NAME" ]]; then
  ARGS+=(--data_name "$DATA_NAME")
fi
if [[ -n "$SPLIT" ]]; then
  ARGS+=(--split "$SPLIT")
fi
if [[ -n "$START_IDX" ]]; then
  ARGS+=(--start_idx "$START_IDX")
fi
if [[ -n "$END_IDX" ]]; then
  ARGS+=(--end_idx "$END_IDX")
fi

"$PYTHON_BIN" "$REPO_DIR/eval/plot_section_3_4_3.py" "${ARGS[@]}"
