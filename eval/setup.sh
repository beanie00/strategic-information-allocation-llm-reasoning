pip install uv
uv venv --python 3.12
source .venv/bin/activate
uv pip install -U -r requirements.txt --torch-backend auto
uv pip install --no-build-isolation flash_attn
