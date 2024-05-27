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
            #pre_cmd="python ./llm-jp-sft/train.py"

            cmd = f"""{pre_cmd}
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