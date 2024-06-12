import glob
import os
import random
import argparse
import time


lr_list = [
     "5e-5",
]

model_name_list=[
"/storage5/llm/models/hf/step62160_fin",
]


# argparseのパーサーを作成
parser = argparse.ArgumentParser(description='引数を取るサンプルスクリプト')
parser.add_argument('data_dir', type=str, help='1')
parser.add_argument('job_name', type=str, help='2')

args = parser.parse_args()
job_name=args.job_name

data_dir=args.data_dir
data_dir='/storage5/EvalPractice/3_finetune/data/0524with_halcination_little_codes_synth_eng'
inst_path_list = (glob.glob(f"{data_dir}/*.parquet"))

#evalは抜く
inst_path_list=[i for i in inst_path_list if i.find("_eval.parquet")==-1]
print(inst_path_list)

random.shuffle(model_name_list)


for model_name in model_name_list:
    print("train: ",model_name)
    for inst_path in inst_path_list:
        for lr in lr_list:
            out_name = job_name+"_"+model_name+"_inst_"+inst_path
            out_name = out_name.replace(".jsonl", "").replace(
                "/", "-").replace(".", "-").replace("data-", "")
            out_name = out_name+"_lr_"+lr
            out_path = "../model/"+out_name
            eval_path = inst_path.replace(".parquet","_eval.parquet")

            print(eval_path)
            print(model_name)
            print(out_path)
            
            #if os.path.exists(out_path):
            #    print("aldeady done")
            #    continue


            #マルチgpu
            pre_cmd="accelerate launch --config_file ./llm-jp-sft/configs/accelerate_config_zero1.yaml ./llm-jp-sft/train.py"
            #通常
            cmd = f"""{pre_cmd}  \
                --num_train_epochs 3 \
                --per_device_train_batch_size 5 \
                --per_device_eval_batch_size 3 \
                --save_strategy "steps" \
                --save_steps 1000 \
                --logging_steps 1 \
                --gradient_accumulation_steps 16 \
                --learning_rate {lr} \
                --warmup_ratio 0.1 \
                --lr_scheduler_type cosine \
                --bf16 \
                --data_files {inst_path} \
                --model_name_or_path {model_name} \
                --use_fast True \
                --output_dir {out_path} \
                --instruction_template "\n\n### 指示:\n" \
                --response_template "\n\n### 応答:\n" \
                --use_flash_attention_2 True \
                --gradient_checkpointing true \
                --max_seq_length 4096 \
                --eval_data_files {eval_path} \

            """

 

            #--load_in_4bit True \
            os.system(cmd)
# --response_template "\n\n### 応答:\n" \
"""
                --peft_target_model mixtral \
                --use_peft True \
                --peft_lora_r 4096 \
                --peft_lora_alpha 4096 \
"""