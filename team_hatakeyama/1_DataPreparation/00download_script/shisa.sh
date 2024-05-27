#shisa
cd ../data/original_dump
wget https://huggingface.co/datasets/augmxnt/shisa-pretrain-en-ja-v1/resolve/main/dataset.parquet
cd ../../00download_script

conda activate textprocess
python shisa.py
