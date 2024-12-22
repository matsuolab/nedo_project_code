
# 学習コード全般


# Folder index
## [hatakeyama](hatakeyama)
    - 事前学習(8b, 8x8b)
    - データ合成
    - [technical data](./technical_data.xlsx)
      - 学習に用いたコーパス一覧､ハイパーパラメータの最適化､シェルスクリプトの一覧など

## [sftlab](sftlab)
- 事後学習(SFT)(8b, 8x8b)
  
## [polab](polab)
- 事後学習(DPO)(8b, 8x8b)

## [ota](ota)
- データ合成
  - [Iterative DPO, LLM-as-a-Judge](ota/iterative-dpo)
    - 合成したプリファレンスデータセット (DPO/RLHF用)
      - [synth-dpo-basic-reasoning-nemotron-4-raw](https://huggingface.co/datasets/team-hatakeyama-phase2/synth-dpo-basic-reasoning-nemotron-4-raw)
      - [synth-dpo-basic-reasoning-nemotron-4](https://huggingface.co/datasets/team-hatakeyama-phase2/synth-dpo-basic-reasoning-nemotron-4)
      - [synth-dpo-basic-math-nemotron-4-raw](https://huggingface.co/datasets/team-hatakeyama-phase2/synth-dpo-basic-math-nemotron-4-raw)
      - [synth-dpo-basic-math-nemotron-4](https://huggingface.co/datasets/team-hatakeyama-phase2/synth-dpo-basic-math-nemotron-4)
  - [Topic-Hub, Persona-Hub](ota/topic-hub)
    - 合成したSFTデータセット
      - [synth-topic-jp-basic-math-calm3](https://huggingface.co/datasets/team-hatakeyama-phase2/synth-topic-jp-basic-math-calm3)
      - [synth-topic-jp-basic-reasoning-calm3](https://huggingface.co/datasets/team-hatakeyama-phase2/synth-topic-jp-basic-reasoning-calm3)
      - [synth-topic-jp-reasoning-nemotron-4](https://huggingface.co/datasets/team-hatakeyama-phase2/synth-topic-jp-reasoning-nemotron-4)
      - [synth-topic-jp-reasoning-calm3](https://huggingface.co/datasets/team-hatakeyama-phase2/synth-topic-jp-reasoning-calm3)
      - [synth-topic-jp-highschoolmath-nemotron-4](https://huggingface.co/datasets/team-hatakeyama-phase2/synth-topic-jp-highschoolmath-nemotron-4)
      - [synth-persona-jp-math-nemotron-4](https://huggingface.co/datasets/team-hatakeyama-phase2/synth-persona-jp-math-nemotron-4)
      - [synth-persona-jp-reasoning-nemotron-4](https://huggingface.co/datasets/team-hatakeyama-phase2/synth-persona-jp-reasoning-nemotron-4)
      - [synth-topic-jp-coding-calm3](https://huggingface.co/datasets/team-hatakeyama-phase2/synth-topic-jp-coding-calm3)
  - [Magpie](ota/magpie)
    - 合成したSFTデータセット
      - [synth-magpie-jp-math-nemotron-4](https://huggingface.co/datasets/team-hatakeyama-phase2/synth-magpie-jp-math-nemotron-4)
      - [synth-magpie-jp-coding-nemotron-4](https://huggingface.co/datasets/team-hatakeyama-phase2/synth-magpie-jp-coding-nemotron-4)
      - [synth-magpie-jp-reasoning-nemotron-4](https://huggingface.co/datasets/team-hatakeyama-phase2/synth-magpie-jp-reasoning-nemotron-4)
  - [ルールベース](ota/rule-based)
    - 合成した事前学習データセット
      - [synth-rule-arithmetic-qa](https://huggingface.co/datasets/team-hatakeyama-phase2/synth-rule-arithmetic-qa)
- データに対するスコア付け
  - [Ask-LLM](ota/ask-llm)
  - [言語判定](ota/lang-identifier)
- Nemotron-4 推論環境構築
  - [vLLM FP8 量子化](ota/nemotron-vllm-fp8)
  - [Megatron](ota/nemotron-megatron)

## [Pre Training Data](pre_training_data)
- 事前学習データを取得するためのコード、フィルタなどを格納しています
  - ndlj
    - 国会図書館デジタルコレクションから著作権切れのデータを取得するコード

## [arata](arata)
- vLLMを用いたデータ合成用の簡易スクリプト群
  - Magpieによるinstructionの作成
  - Evol-Instructによるinstructionの作成
    - [Hajime-Y/Evol-Instruct-jp](https://github.com/Hajime-Y/Evol-Instruct-jp)のプロンプトを参考に作成
  - バッチ推論による応答の作成

## [llm-leaderboard](llm-leaderboard)
- 評価
  - Nejumi-leaderboard Neo for Tanuki
  - JMT-Benchのみ、llm-jp-evalのみ、その両方の評価が可能

# ライセンス
- ライセンスはMITとなります(一部､他のレポジトリからcloneしたコードが含まれるフォルダが存在します｡そのフォルダ内のライセンスは､元のライセンスに従います)｡
