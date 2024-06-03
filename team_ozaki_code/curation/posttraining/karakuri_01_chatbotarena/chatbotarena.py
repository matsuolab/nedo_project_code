import os, datetime, time, json
import pandas as pd
import torch
from datetime import timezone, timedelta
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

def try_multiple_encodings(file_path):
    encodings = ['utf-8', 'shift_jis', 'latin1', 'iso-8859-1', 'cp1252']
    for enc in encodings:
        try:
            return pd.read_csv(file_path, encoding=enc)
        except:
            print(f"Failed with encoding {enc}")
    raise ValueError("None of the encodings worked.")


# タイムゾーンの設定
JST = timezone(timedelta(hours=+9))

long_token_file_path = os.path.join('/home/user', 'long_token_chatbot.json')

# フォルダパスの指定
input_folder = '/home/user/miniconda3/work/data'

# モデルとトークナイザのロード
model = AutoAWQForCausalLM.from_quantized(
    "GENIAC-Team-Ozaki/karakuri-lm-8x7b-chat-v0.1-awq",
    fuse_layers=True,
    trust_remote_code=False,
    safetensors=True,
)

tokenizer = AutoTokenizer.from_pretrained(
    "GENIAC-Team-Ozaki/karakuri-lm-8x7b-chat-v0.1-awq",
    trust_remote_code=False,
)

# 入力フォルダ内のすべてのCSVファイルを処理
for filename in os.listdir(input_folder):
    if filename.endswith('.csv'):
        csv_file_path = os.path.join(input_folder, filename)
        json_filename = filename.replace('.csv', '.json')
        json_file_path = os.path.join(input_folder, json_filename)

        # CSVファイルの読み込み
        df = try_multiple_encodings(csv_file_path)
        # df = pd.read_csv(csv_file_path)
        # annotationが1または2のデータのみ抽出
        df = df[df['annotation'].isin([1, 2])]

        # CSVデータの処理
        for i, row in df.iterrows():
            index = row['index']
            prompt = row['prompt']
            chosen = row['chosen']
            rejected = row['rejected']
            annotation = row['annotation']
            messages = [{"role": "user", "content": prompt}]
            input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").cuda()
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            attention_mask = input_ids.ne(tokenizer.pad_token_id).int()
            start_time = time.time()
            outputs = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=1024, do_sample=True, temperature=0.99, top_p=0.95)
            end_time = time.time()
            elapsed_time = end_time - start_time
            generated_text = tokenizer.decode(outputs[0][input_ids.shape[-1]:])
            if '</s>' in generated_text:
                generated_text = generated_text.replace('</s>', '')
                data = {
                    'index': index,
                    'prompt': prompt,
                    'chosen': generated_text,
                    'rejected': rejected,
                    'original_chosen': chosen,
                    'annotation': annotation, 
                    'time': f"{elapsed_time:.2f}"
                }
                # JSONファイルにデータを追記
                with open(json_file_path, mode='a', encoding='utf-8') as file:
                    json.dump(data, file, ensure_ascii=False, indent=4)
            else:
                data = {
                    'index': index,
                    'prompt': prompt,
                    'chosen': generated_text,
                    'rejected': rejected,
                    'original_chosen': chosen,
                    'annotation': annotation, 
                    'time': f"{elapsed_time:.2f}"
                }
                with open(long_token_file_path, mode='a', encoding='utf-8') as long_file:
                    json.dump(data, long_file, ensure_ascii=False, indent=4)
            print(f"Index: {index}, Prompt: {prompt}, 処理時間: {elapsed_time:.2f}秒")
