conda create -n llava-factory python=3.10 -y
conda activate llava-factory
pip install -e .
pip install flash-attn==2.6.3
pip install boto3