python plot_mi.py --csv2 /workspace/2026-LIMO/mi_peak/results2/limo-qwen3-8b/mi_first2500_layer36.csv --csv /workspace/2026-LIMO/mi_peak/results2/base-qwen3-8b/mi_first2500_layer36.csv \
    --title "Qwen3-8B-Base" \
    --title2 "Qwen3-8B-Base-SFT" \
    --output comparison

python plot_mi.py --csv /workspace/2026-LIMO/mi_peak/results2/limo-qwen3-8b/mi_first2500_layer36.csv --title "Natural Reasoning" --output natural

python plot_mi.py --csv /workspace/2026-LIMO/mi_peak/results2/base-deepseek-32b/mi_first5000_layer64.csv --title "Natural Reasoning" --output natural_deepseek

