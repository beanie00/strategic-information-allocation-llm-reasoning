#!/bin/bash

# GPU assignment
GENERATOR_GPUS="0,1,2,3"
EVALUATOR_GPUS="4,5,6,7"

# Model paths
GENERATOR_MODEL="Qwen/Qwen3-14B-Base"
EVALUATOR_MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"

# Port assignment
GENERATOR_PORT=8001
EVALUATOR_PORT=8010

echo "Starting SGLang servers..."
echo ""

# Start generator server
echo "Starting Generator: $GENERATOR_MODEL on GPUs $GENERATOR_GPUS (Port $GENERATOR_PORT)"
CUDA_VISIBLE_DEVICES=$GENERATOR_GPUS python -m sglang.launch_server \
  --model-path $GENERATOR_MODEL \
  --port $GENERATOR_PORT \
  --host 0.0.0.0 \
  --tp 2 \
  --trust-remote-code \
  --mem-fraction-static 0.85 \
  > generator_server.log 2>&1 &

GENERATOR_PID=$!
echo "  Generator PID: $GENERATOR_PID"
echo "  Log: generator_server.log"
echo ""

sleep 5

# Start evaluator server
echo "Starting Evaluator: $EVALUATOR_MODEL on GPUs $EVALUATOR_GPUS (Port $EVALUATOR_PORT)"
CUDA_VISIBLE_DEVICES=$EVALUATOR_GPUS python -m sglang.launch_server \
  --model-path $EVALUATOR_MODEL \
  --port $EVALUATOR_PORT \
  --host 0.0.0.0 \
  --tp 2 \
  --trust-remote-code \
  --mem-fraction-static 0.85 \
  > evaluator_server.log 2>&1 &

EVALUATOR_PID=$!
echo "  Evaluator PID: $EVALUATOR_PID"
echo "  Log: evaluator_server.log"
echo ""

echo $GENERATOR_PID > generator.pid
echo $EVALUATOR_PID > evaluator.pid

echo "Both servers are starting..."
echo ""
echo "To monitor logs:"
echo "  tail -f generator_server.log evaluator_server.log"