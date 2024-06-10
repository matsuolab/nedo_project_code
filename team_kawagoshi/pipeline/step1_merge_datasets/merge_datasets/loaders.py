from datasets import load_dataset

def refinedweb_en_loader(streaming=True, split='train'):
    return load_dataset("tiiuae/falcon-refinedweb", split=split,
                        streaming=streaming
                        )

def slimpajama_en_loader(streaming=False, split='train[0%:1%]'):
    return load_dataset("cerebras/SlimPajama-627B", split=split, 
                        streaming=streaming
                        )

def test_wiki_ja_loader(streaming=True, split='train'):
    return load_dataset("kawagoshi-llm-team/test_preprocessed_wiki_dataset_gz", split=split,
                        streaming=streaming
                        )

def test_mc4_ja_loader(streaming=True, split='train'):
    return load_dataset("kawagoshi-llm-team/test_preprocessed_mc4_dataset", split=split, 
                        streaming=streaming
                        )