from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import shutil

home_directory = os.getenv('HOME')
#model_dir = os.path.join(home_directory, "tokenizer_model_directory")
model_dir = "/storage7/tokenizer_model_directory"
# ディレクトリが存在する場合は削除
if os.path.exists(model_dir):
    shutil.rmtree(model_dir)

# ディレクトリを新規作成
os.makedirs(model_dir)

tokenizer = AutoTokenizer.from_pretrained("geniacllm/ja-en-tokenizer-unigram-v5")
tokenizer.save_pretrained(model_dir)
