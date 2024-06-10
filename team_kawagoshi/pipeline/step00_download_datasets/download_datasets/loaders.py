from datasets import load_dataset


def wiki_ja_loader(streaming=True, split='train'):
    return load_dataset("hpprc/wikipedia-20240101", split=split,
                        streaming=streaming
                        ).shuffle()

def mc4_ja_part_loader(streaming=True, split='train'):
    return load_dataset("izumi-lab/mc4-ja", split=split,
                        # data_files="data/train-00000-of-00416-a74a40664a952804.parquet",
                        streaming=streaming
                        )

def refinedweb_en_loader(streaming=True, split='train'):
    return load_dataset("tiiuae/falcon-refinedweb", split=split,
                        streaming=streaming
                        )

def slimpajama_en_loader(streaming=False, split='train[0%:1%]'):
    return load_dataset("cerebras/SlimPajama-627B", split=split, 
                        streaming=streaming
                        )