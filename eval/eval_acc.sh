nohup bash -c '
for model in "[path]/Qwen3-4B-Base_sft" "[path]/Qwen3-8B-Base_sft"; do
  model_short=$(echo "$model" | sed "s|.*/||")
  for data in aime25 aime amc math; do
    VLLM_ATTENTION_BACKEND=FLASH_ATTN CUDA_VISIBLE_DEVICES="0,1,2,3" \
    python eval_acc.py \
      --model_name_or_path "$model" \
      --data_name "$data" \
      --prompt_type "qwen-instruct" \
      --temperature 0.7 \
      --start_idx 0 \
      --end_idx -1 \
      --n_sampling 16 \
      --split "test" \
      --max_tokens 32768 \
      --seed 0 \
      --top_p 1.0 \
      --output_dir "./outputs_${model_short}/${data}" \
      --surround_with_messages
  done
done
' > eval_log.out 2>&1 &