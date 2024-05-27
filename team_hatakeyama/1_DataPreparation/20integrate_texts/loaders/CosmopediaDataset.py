import random
from datasets import load_dataset


class CosmopediaDataset:
    def __init__(self, streaming=True):
        self.dataset = load_dataset(
            "kunishou/cosmopedia-100k-ja-preview", split="train").shuffle()
        self.loader = iter(self.dataset)

    def __iter__(self):
        # イテレータは自分自身を返す
        return self

    def __next__(self):
        # ランダムな順序で日英を返す
        d = next(self.loader)
        if random.random() < 0.5:
            d["text"] = d["text_ja"]+"\n"+d["text"]
        else:
            d["text"] = d["text"]+"\n"+d["text_ja"]
        return d
