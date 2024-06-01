import sentencepiece as spm

def load(model_file):
    return spm.SentencePieceProcessor(model_file=model_file)
