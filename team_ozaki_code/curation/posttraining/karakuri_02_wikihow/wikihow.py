import os, time, json
import pandas as pd
import torch
from datetime import timezone, timedelta
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
from datasets import load_dataset

def try_multiple_encodings(file_path):
    encodings = ['utf-8', 'shift_jis', 'latin1', 'iso-8859-1', 'cp1252']
    for enc in encodings:
        try:
            return pd.read_csv(file_path, encoding=enc)
        except:
            print(f"Failed with encoding {enc}")
    raise ValueError("None of the encodings worked.")

def generate_response(model, tokenizer, input_ids, attention_mask, max_new_tokens):
    start_time = time.time()
    outputs = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.99, top_p=0.95)
    end_time = time.time()
    elapsed_time = end_time - start_time
    generated_text = tokenizer.decode(outputs[0][input_ids.shape[-1]:])
    return generated_text, elapsed_time

dataset = load_dataset("GENIAC-Team-Ozaki/WikiHowNFQA-ja_cleaned")

# タイムゾーンの設定
JST = timezone(timedelta(hours=+9))

json_file_path = os.path.join('/home/user', 'all_data_wiki.json')
long_token_file_path = os.path.join('/home/user', 'long_token_wiki.json')

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

for i in range(len(dataset['train'])):
    entry = dataset['train'][i]
    prompt = entry["question"]
    messages = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").cuda()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    attention_mask = input_ids.ne(tokenizer.pad_token_id).int()

    generated_text, elapsed_time = generate_response(model, tokenizer, input_ids, attention_mask, 1024)

    if '</s>' not in generated_text:
        generated_text, elapsed_time = generate_response(model, tokenizer, input_ids, attention_mask, 2048)

    if '</s>' in generated_text:
        generated_text = generated_text.replace('</s>', '')
        data = {
            "article_id": entry["article_id"],
            "question": prompt,
            "answer": entry["answer"],
            "new_answer": generated_text,
            'time': f"{elapsed_time:.2f}"
        }
        # JSONファイルにデータを追記
        with open(json_file_path, 'a', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    else:
        data = {
            "article_id": entry["article_id"],
            "question": prompt,
            "answer": entry["answer"],
            "new_answer": generated_text,
            'time': f"{elapsed_time:.2f}"
        }
        with open(long_token_file_path, 'a', encoding='utf-8') as long_file:
            json.dump(data, long_file, ensure_ascii=False, indent=4)

    print(f"Index: {entry['article_id']}, Prompt: {prompt}, 処理時間: {elapsed_time:.2f}秒")

print("finished")
