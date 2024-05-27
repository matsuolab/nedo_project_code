
cd ../data/original_dump
mkdir commoncrawl

GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/hatakeyama-llm-team/CommonCrawl_wet_v2

cd ../../00download_script/
conda activate textprocess
python CommonCrawlWetv2.py