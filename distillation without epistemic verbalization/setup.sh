pip install uv
uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install -U vllm==0.11.0 --torch-backend auto
uv pip install "sglang[all]"