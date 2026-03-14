conda create -n llama-factory python=3.11 -y
conda activate llama-factory

git clone --depth 1 https://github.com/hiyouga/LlamaFactory.git
cd LlamaFactory
pip install -e ".[torch,metrics]"
cd ../

pip install "deepspeed>=0.10.0,<=0.16.9"
pip install flash-attn