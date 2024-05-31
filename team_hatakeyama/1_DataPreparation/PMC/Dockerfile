FROM python:3.9.15

RUN apt-get update && apt-get install -y \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN pip install poetry==1.8.2

WORKDIR /app

COPY pyproject.toml poetry.lock* /app/
COPY src/ /app
COPY target/ /app/target
COPY --from=apache/beam_python3.9_sdk:2.54.0 /opt/apache/beam /opt/apache/beam

# Poetryを使って依存関係をインストール
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

ENTRYPOINT ["/opt/apache/beam/boot"]
