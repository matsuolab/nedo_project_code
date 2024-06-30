## トークナイザーの学習

### 学習の実行

```
cd ~/nedo_project_code/team_kawagoshi/pipeline
```
llm-jp-tokenizerをクローン。
```
git clone https://github.com/llm-jp/llm-jp-tokenizer.git
```

scriptsへ移動。
```
cd ~/nedo_project_code/team_kawagoshi/pipeline/llm-jp-tokenizer/scripts
```

スコアの再推定に利用するため、multigramをクローン。
```
git clone https://github.com/tatHi/multigram.git
```

下記、markdownに従って作業を実施。
~/nedo_project_code/team_kawagoshi/pipeline/llm-jp-tokenizer/scripts/howToCreateModel_ver2.md


sentencepieceによるトークナイズ学習の際は、下記を実行。
```sh
(.venv) $ cd ~/nedo_project_code/team_kawagoshi/pipeline/step3_tokenization

# 学習スクリプトを実行。
(.venv) $ python .train_tokenizer/train_sentencepiece_tokenizer.py \
    --input ~/nedo_project_code/team_kawagoshi/pipeline/step3_tokenization/output/botchan.model \
    --output_base ./output \
    --model_prefix botchan \
    --vocab_size 2000
```

### トークナイザーのHuggingFace Hubへのアップロード

```sh
(.venv) $ cd ~/nedo_project_code/team_kawagoshi/pipeline/step3_tokenization/upload_tokenizer

# HuggingFaceにログインしていることを確認。
(.venv) $ huggingface-cli whoami

# アップロードスクリプトを実行。
(.venv) $ python ./upload_tokenizer_and_finetuned_model_to_huggingface_hub.py \
    --input_tokenizer_file ~/ucllm_nedo_dev/train/output/step4_finetune_model/gpt_0.125B_global_step1000_openassistant/ \
    --output_model_name gpt_0.125B_global_step1000_openassistant \
    --test_prompt_text "Once upon a time,"
```
