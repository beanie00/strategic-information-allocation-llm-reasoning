pip install uv
uv venv --python 3.12
source .venv/bin/activate
uv pip install -U vllm --torch-backend auto