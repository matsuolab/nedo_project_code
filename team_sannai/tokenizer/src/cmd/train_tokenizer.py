from src.preprocess.preprocess import preprocess_financial_economic_text, preprocess_latex_math, preprocess_legal_contract_text, preprocess_mathematical_text, preprocess_medical_scientific_text, preprocess_natural_language_text, preprocess_social_media_text
from src.tokenizer.train import train

def train_tokenizer(input_file_path, category, vocab_size=20000, coverage=0.995, model_type='bpe', max_sentence_length=4096):
    with open(input_file_path, 'r') as f:
        content = f.read()

    removed_elements = []

    if category == 'natural':
        preprocessed_text, removed = preprocess_natural_language_text(content)
    elif category == 'news':
        preprocessed_text, removed = preprocess_natural_language_text(content)
    elif category == 'book':
        preprocessed_text, removed = preprocess_natural_language_text(content)
    elif category == 'medical':
        preprocessed_text, removed = preprocess_medical_scientific_text(content)
    elif category == 'financial':
        preprocessed_text, removed = preprocess_financial_economic_text(content)
    elif category == 'social':
        preprocessed_text, removed = preprocess_social_media_text(content)
    elif category == 'legal':
        preprocessed_text, removed = preprocess_legal_contract_text(content)
    elif category == 'math':
        preprocessed_text, removed = preprocess_mathematical_text(content)
        preprocessed_text, latex_removed = preprocess_latex_math(preprocessed_text)
        removed.extend(latex_removed)
    else:
        print(f"Unsupported genre: {category}")
        return

    removed_elements.extend(removed)
    print(f"removed_elements are {removed_elements}")

    output_file_path = f"./artifacts/preprocessed/{category}.txt"
    with open(output_file_path, 'w') as f:
        f.write(preprocessed_text)
    print(f"Preprocessed text has been saved to {output_file_path}")

    # Train tokenizer
    train([output_file_path], category, vocab_size=vocab_size, coverage=coverage, model_type=model_type, max_sentence_length=max_sentence_length)
