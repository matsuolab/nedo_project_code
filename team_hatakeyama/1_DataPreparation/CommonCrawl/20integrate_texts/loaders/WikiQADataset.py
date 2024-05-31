
import random
from datasets import load_dataset


class WikiQADataset:
    def __init__(self, streaming=True):
        self.dataset = load_dataset(
            "alfredplpl/wikipedia-qa-ja-1m", split="train").shuffle()
        self.dataset = self.dataset.filter(lambda x: x["answer"][-1] == "。")

        self.loader = iter(self.dataset)

    def __iter__(self):
        # イテレータは自分自身を返す
        return self

    def __next__(self):
        # ランダムな順序で日英を返す
        d = next(self.loader)
        d["text"] = d["question"]+"\n"+d["answer"]
        return d
