# Pretrain：はじめに
**WARNING: このREADMEは工事中です！（最終更新日: 2024/5/18）**

このリポジトリはMoE（Mixture of Experts）モデルの事前学習のためのリポジトリです．
## 環境構築
environment_guide.mdの指示に従い，環境構築をしてください．
## 事前学習の実行
デフォルトではDeepSeekMoEアーキテクチャのモデルを学習するようになっています．


---以下工事中---
### 仮想環境の起動
loginノードで，```conda activate <env_name>```を実行する．
### モデルの初期化（初回のみ）

### configの設定
### データの処理（tokenize, grouping）
```sbatch tokenize.sh```
### 事前学習の実行
```sbatch multi.sh```



## Checkpoint conversion
チェックポイントができたら，これらをHugging Faceにアップロードできます．

srunでシングルノードシングルGPUに入る．
```
deepspeed --no_local_rank pretrain/utils/ckpt_upload.py --ckpt-path <ckpt-path> --repo-name <repo-name>
```
