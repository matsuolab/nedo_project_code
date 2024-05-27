#cc100
cd ../data/original_dump
wget https://data.statmt.org/cc-100/ja.txt.xz
cd ../../download_script
conda activate textprocess
python cc.py