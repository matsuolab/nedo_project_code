
import random
from datasets import load_dataset
import os

from datasets import load_dataset
with open(".env", "r") as f:
    for line in f:
        var = line.split("=")
        os.environ[var[0]] = var[1].strip()


class NHKSchoolDataset:
    def __init__(self):
        self.dataset = load_dataset("hatakeyama-llm-team/nhk_for_school_outline",
                                    split="train",
                                    use_auth_token=os.environ["hf_key"],
                                    )

        self.dataset = self.dataset.filter(
            lambda x: (x["outline"]) is not None)
        self.loader = iter(self.dataset)

    def __iter__(self):
        # イテレータは自分自身を返す
        return self

    def __next__(self):
        # ランダムな順序で日英を返す
        d = next(self.loader)
        text = d["outline"]
        text = clean_text(text)
        d["text"] = text
        return d


def clean_text(text):
    lines = text.split("\n")

    new_lines = []
    for line in lines:
        if line in new_lines:
            continue
        if line.strip() == "":
            continue
        if line.find("【") >= 0:
            continue
        if line.find("オープニング") >= 0:
            continue

        new_lines.append(line)

    return "\n".join(new_lines)
