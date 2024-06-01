import csv

from src.cmd.prepare_data import  prepare_data as cmd_prepare_data
from src.cmd.train_tokenizer import train_tokenizer as cmd_train_tokenizer
from src.cmd.evaluate_tokenizer import evaluate_tokenizer as cmd_evaluate_tokenizer

def prepare_data(key, category):
    input_path = f"./corpus/{category}"
    print(f"Loading data from {input_path}")
    cmd_prepare_data(input_path, key, category)

def train_tokenizer(category, vocab_size, coverage, model_type, max_sentence_length):
    input_file_path = f"./artifacts/prepared/{category}.txt"
    print(f"Train tokenizer for {category}, input_file_path={input_file_path}")
    cmd_train_tokenizer(input_file_path, category, vocab_size, coverage, model_type, max_sentence_length)

def evaluate(category, eval_data_path):
    model_file_path=f"./artifacts/tokenizers/custom_tokenizer-{category}.model"
    print(f"Evauate tokenizer for {model_file_path}. data_path={eval_data_path}")
    return cmd_evaluate_tokenizer(model_file_path, eval_data_path)

if __name__ == '__main__':
    csv_file_path = './config/setting.csv'

    with open("./artifacts/eval_result/result.csv", mode='w', newline='', encoding='utf-8') as file:
        # 辞書のキーからフィールド名を取得し、CSVのヘッダーとして使用
        fieldnames = ['id', 'category', 'vocab_size', 'coverage', 'model_type', 'max_sentence_length', 'eval_data_path', 'unk_rate', 'avg_len', 'speed', 'error_word_count']
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        # ヘッダーを書き込む
        writer.writeheader()

        with open(csv_file_path, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                id = row['id']
                category = row['category']
                vocab_size = int(row['vocab_size'])
                coverage = float(row['coverage'])
                model_type = row['model_type']
                max_sentence_length= int(row['max_sentence_length'])
                eval_data_path=row['eval_data_path']
                print(f"exec with id:{id}, category:{category}, vocab_size:{vocab_size}, coverage:{coverage}, model_type:{model_type}")
                prepare_data(key='text', category=category)
                train_tokenizer(category=category, vocab_size=vocab_size, coverage=coverage, model_type=model_type, max_sentence_length=max_sentence_length)
                eval_result = evaluate(category=category, eval_data_path=eval_data_path)
                eval_result['id'] = id
                eval_result['category'] = category
                eval_result['vocab_size'] = vocab_size
                eval_result['coverage'] = coverage
                eval_result['model_type'] = model_type
                eval_result['max_sentence_length'] = max_sentence_length
                eval_result['eval_data_path'] = eval_data_path
                error_words = eval_result.pop('error_words')
                writer.writerow(eval_result)

                with open(f"./artifacts/eval_result/error_words/{id}.csv", 'w', newline='', encoding='utf-8') as f:
                    w = csv.writer(f)
                    w.writerow(['error_word'])
                    w.writerows([[word] for word in error_words])
