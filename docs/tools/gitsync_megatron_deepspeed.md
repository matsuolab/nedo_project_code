# Megatron-DeepSpeedのGit同期

## 概要

- trainフォルダのMegatron-DeepSpeedと以下のリポジトリについて同期方法
  - 本家Megatron-DeepSpeed
  - 標準コードとされるhotsuyuki/Megatron-DeepSpeed
- 現時点では差分はないため、実際の更新方法については未記載
- 本家Megatron-DeepSpeedのリポジトリは以下
  - https://github.com/microsoft/Megatron-DeepSpeed
- hotsuyki版のリポジトリは以下
  - https://github.com/hotsuyuki/Megatron-DeepSpeed

## 同期方法

### 本家リポジトリ

***2024/04/20現在、差分があるが、修正内容がコンフリクトするためマージしない。コンフリクトしない他のファイルについても本プロジェクトとは関連しない箇所なのでマージしていない***

1. 本家リポジトリをCloneする。(tmpフォルダはgitignoreに記載されているため、tmp/microsoftフォルダ内にCloneする)
  - 既にCloneしている場合は、`git pull`などで最新の情報を取得する。

```bash
git clone https://github.com/microsoft/Megatron-DeepSpeed.git tmp/microsoft/Megatron-DeepSpeed
```

2. ローカルのソースコードのバージョンを前回同期時のタグに変更する。check/20240325は例。任意のブランチ名を指定する。前回作成しているのであれば、そのブランチに切り替えるのでも可。（`git switch check/20240325`）

```bash
git checkout -b check/20240325 refs/tags/megatron-deepspeed-20240325
```

3. 差分比較を行う。

```bash
diff -rx .git train/Megatron-DeepSpeed tmp/microsoft/Megatron-DeepSpeed
```

* 差分が出た場合には基本的にはtrainフォルダにコピーする事になるが、それに伴い動かなくなるケースやコンフリクトのケースも想定されることから手順としては、実際に差分が発生した際に検討する。

### hotsuyuki版リポジトリ

1. hotsuyuki版リポジトリをCloneする。(tmpフォルダはgitignoreに記載されているため、tmp/hotsuyukiフォルダ内にCloneする)
  - 既にCloneしている場合は、`git pull`などで最新の情報を取得する。

```bash
git clone https://github.com/hotsuyuki/Megatron-DeepSpeed tmp/hotsuyuki/Megatron-DeepSpeed
```

2. hotsuyuki版はタグが振られているため、最新のタグを確認する。

```bash
cd tmp/hotsuyuki/Megatron-DeepSpeed
git tag
```

3. タグの切り替え

```bash
cd tmp/hotsuyuki/Megatron-DeepSpeed
git checkout -b check/20240325 refs/tags/ucllm_nedo_dev_v20240415.1.0
```

4. 差分比較を行う。

```bash
cd ../../..
diff -rx .git train/Megatron-DeepSpeed tmp/hotsuyuki/Megatron-DeepSpeed
```
