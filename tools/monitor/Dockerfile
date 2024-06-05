FROM python:3.11-slim-bullseye

# curl等必要コンポーネントのインストール
RUN apt-get update && apt-get install -y curl jq apt-transport-https ca-certificates gnupg openssh-client

# debian版gcloud cliのインストール
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list \
    && apt-get update && apt-get install -y google-cloud-cli

# ワークディレクトリの設定
WORKDIR /app

# 実行ファイルのコピー
COPY exec.sh /app/exec.sh
COPY monitor.py /app/monitor.py

# 実行権限の付与
RUN chmod +x /app/exec.sh

# 実行
CMD ["./exec.sh"]
