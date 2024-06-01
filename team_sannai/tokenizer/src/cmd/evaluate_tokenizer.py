from src.tokenizer.load import load
from src.eval.metric import evaluate_tokenizer as eval

def evaluate_tokenizer(model_file, data_path):
    tokenizer = load(model_file)
    with open(data_path, 'r') as f:
        content = f.read()

    return eval(tokenizer, content)