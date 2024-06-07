# SentencePieceモデルからLlamaTokenizer形式への変換
## 参考：松尾研標準コード
- https://github.com/matsuolab/ucllm_nedo_prod/blob/main/train/scripts/step3_upload_pretrained_model/convert_tokenizer_from_sentencepiece_to_huggingface_transformers.py

## 準備
- スコア再推定後のSentencePieceモデルを適当なディレクトリに置いておく。

## 実行
```python
python -m convert_tokenizer_from_sentencepiece_to_huggingface_transformers \
--input_tokenizer_file "models/sample_new.model" \
--output_tokenizer_dir "models/hf/"
```
- 実行すると以下のファイルが出力される
    - special_tokens_map.json
    - spiece.model
    - tokenizer_config.json
