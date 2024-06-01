# Env

used Python 3.10.4

setup example

```
pyenv local 3.10.4
python -m venv myenv
```

# Setup

```
pip install -r requirements.txt
```

# Run experiment

write csv in `./config/setting.csv`

place data in `./corpus` folder

Make sure to create subfoler named with `category` specified in `setting.csv`

Below is example of how to setup corpus. jsonl file name can be any name as long as its extension is `.jsonl`. multiple `.jsonl` files can be placed.

```
├── corpus
│   └── natural
│       └── livedoornews.jsonl
```

Run below command to train and calc metric for tokenizer

```
make run
```

Result is placed as `./artifacts/eval_result/result.csv`