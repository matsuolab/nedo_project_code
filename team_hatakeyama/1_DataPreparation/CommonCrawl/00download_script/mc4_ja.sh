#mc4-ja
cd ../data../original_dump
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/allenai/c4
cd c4
git lfs pull --include "multilingual/c4-ja.*.json.gz"
