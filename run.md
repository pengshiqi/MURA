source /DATA/data/sqpeng/miniconda2/bin/activate py3
python -m visdom.server
CUDA_VISIBLE_DEVICES="3" python main.py test --load_model_path="
