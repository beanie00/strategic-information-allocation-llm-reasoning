pip install uv
uv venv --python 3.12
source .venv/bin/activate
cd ../LlamaFactory
uv pip install -e ".[torch,metrics]"