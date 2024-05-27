#作ったモデルを動かしてみる
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import time
def perplexity(model, tokenizer, text) -> torch.Tensor:
    tokenized_input = tokenizer.encode(
        text, add_special_tokens=False, return_tensors="pt"
    ).to(model.device)
    with torch.inference_mode():
        output = model(tokenized_input, labels=tokenized_input)
    ppl = torch.exp(output.loss)
    return ppl.item()


model_path="../../models/hf/test"
print("begin loading model")
model = AutoModelForCausalLM.from_pretrained(model_path,device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_path)


pipe=pipeline('text-generation',model=model,tokenizer=tokenizer, max_new_tokens=200, repetition_penalty=1.5)
text_list=["今日はいい",
"富士山は",
"質問: 今日の天気は? 回答:",
"批判的な",
"大規模言語モデルは",
"AI研究の問題点は",
"化学研究の問題点は",
"I have a",
"Functional polymers are",
"機能性高分子は",           
"ホンダ フィット 販売",
]

for text in text_list:
    perp=perplexity(model,tokenizer,text)
    s_time=time.time()
    res=pipe(text)[0]["generated_text"]
    consumed_time=time.time()-s_time
    print("-------")
    print("input: ", text)
    print("perplexity: ",perp)
    print("time: ", consumed_time)
    print("time/character: ", consumed_time/len(res))
    print("output: ",res)

while True:
    text=input()
    res=pipe(text)[0]["generated_text"]
    print(res)