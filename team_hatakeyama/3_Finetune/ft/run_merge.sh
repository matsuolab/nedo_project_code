#!/bin/bash
export PY=/storage5/EvalPractice/3_finetune/0524ft_run_merge.py
export DATA=/storage5/EvalPractice/3_finetune/data/0524with_halcination_little_codes_synth_eng_math
export JOB=/storage5/EvalPractice/3_finetune/0524_5_dataset_clean_halcination_longer_multiturn.py
export PATH=/home/ext_kan_hatakeyama_s_gmail_com/miniconda3/envs/.venv_train/bin:$PATH
export HF_HOME=/storage5/hf
source ~/miniconda3/etc/profile.d/conda.sh && conda activate .venv_train
echo "pyファイル: $PY"
echo "データパス: $DATA"
echo "job名: $JOB"
python $PY $DATA $JOB