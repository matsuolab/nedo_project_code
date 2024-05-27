import os
import random
from datasets import load_dataset


class KokkaiDataset:
    def __init__(self, auth_token):
        self.dataset = load_dataset(
            "JINIAC/ParliamentaryProceedings-filtered",
            use_auth_token=auth_token,
            streaming=True,
            split="train")
        self.loader = iter(self.dataset)

    def __iter__(self):
        # イテレータは自分自身を返す
        return self

    def __len__(self):
        return len(self.dataset)

    def __next__(self):
        d = next(self.loader)
        if "発言内容" in d:
            d["text"] = d["発言内容"]
        else:
            d = ""
        return d


class PythonCodeDataset:
    def __init__(self, streaming=True):
        self.dataset = load_dataset(
            "flytech/python-codes-25k",
            split="train").shuffle()
        self.loader = iter(self.dataset)

    def __iter__(self):
        # イテレータは自分自身を返す
        return self

    def __len__(self):
        return len(self.dataset)

    def __next__(self):
        d = next(self.loader)
        out = d["output"].replace("```", "")
        d["text"] = d["instruction"] + d["input"]+"\n"+out
        d["text"] = d["text"].strip()
        return d


class OpenMathInstructJa:
    def __init__(self, ):
        self.dataset = load_dataset(
            "kunishou/OpenMathInstruct-1-1.8m-ja",
            split="train").shuffle()
        self.loader = iter(self.dataset)

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        # イテレータは自分自身を返す
        return self

    def __next__(self):
        d = next(self.loader)
        j_q = d["question_ja"]
        j_a = d["generated_solution_ja"]
        try:
            txt = j_q+"\n\n"+j_a
        except TypeError:
            return {"text": ""}
        d["text"] = txt
        return d


class WikiBookEn:
    def __init__(self, streaming=True):
        self.dataset = load_dataset(
            "bigscience-data/roots_en_wikibooks",
            split="train", streaming=streaming)
        self.loader = iter(self.dataset)

    def __iter__(self):
        # イテレータは自分自身を返す
        return self

    def __len__(self):
        return len(self.dataset)

    def __next__(self):
        while True:
            d = next(self.loader)
            # UTCが含まれるデータはスキップ｡議論なので｡
            if d["text"].find("UTC") > 0:
                continue
            return d


class PileStackExchange:
    def __init__(self, streaming=True,
                 mode="validation"):
        self.dataset = load_dataset(
            "suolyer/pile_stackexchange",
            split=mode, streaming=streaming)
        self.loader = iter(self.dataset)

    def __iter__(self):
        # イテレータは自分自身を返す
        return self

    def __next__(self):
        d = next(self.loader)
        d["text"] = d["text"].strip()
        return d


class PMCDataset2:
    def __init__(self,
                 streaming=True,
                 mode="train"):
        self.dataset = load_dataset(
            "hatakeyama-llm-team/PMC",
            split=mode, streaming=streaming,
            # """
            # data_files=[
            #    "https://huggingface.co/datasets/hatakeyama-llm-team/PMC/resolve/main/PMC002xxxxxx.jsonl",
            #    "https://huggingface.co/datasets/hatakeyama-llm-team/PMC/resolve/main/PMC001xxxxxx.jsonl",
            #    "https://huggingface.co/datasets/hatakeyama-llm-team/PMC/resolve/main/PMC000xxxxxx.jsonl",
            # ]
        )
        self.loader = iter(self.dataset)

    def __iter__(self):
        # イテレータは自分自身を返す
        return self

    def __next__(self):
        try:
            d = next(self.loader)
            d["text"] = d["text"].replace("Background ", "").strip()
        except:
            return {"text": ""}
        return d


class JaNewsDataset:
    def __init__(self,
                 data_files="https://huggingface.co/datasets/atsushi3110/news-ja/resolve/main/news_cc.jsonl",
                 ):
        self.dataset = load_dataset(
            "atsushi3110/news-ja",
            data_files=data_files,
            split="train",)
        self.loader = iter(self.dataset)

    def __iter__(self):
        # イテレータは自分自身を返す
        return self

    def __next__(self):
        d = next(self.loader)
        t = d["text"]
        noise_words = [
            "(c)", "（抜粋）", "(この記", "関連"
        ]
        for noise in noise_words:
            if t.find(noise) >= 0:
                t = t[:t.find(noise)]

        if t.find("】") >= 0:
            t = t.split("】")[1:]
            t = "".join(t)

        d["text"] = t
        return d


class PMCDataset:
    def __init__(self, streaming=True,
                 ):
        self.dataset = load_dataset(
            "hatakeyama-llm-team/PMC",
            split="train", streaming=streaming)
        self.loader = iter(self.dataset)

    def __iter__(self):
        # イテレータは自分自身を返す
        return self

    def __next__(self):
        d = next(self.loader)
        d["text"] = d["Text"]
        return d


class LightNovelFourM:
    def __init__(self, streaming=True,
                 auth_token=None,
                 ):
        self.dataset = load_dataset("isek-ai/light-novel-4m",
                                    split="train",
                                    use_auth_token=auth_token,
                                    streaming=streaming,
                                    )

        self.loader = iter(self.dataset)

    def __iter__(self):
        # イテレータは自分自身を返す
        return self

    def __next__(self):
        d = next(self.loader)
        d["text"] = d["episode_body"]
        return d


class FlanDataset:
    def __init__(self, ):
        self.dataset = load_dataset("Muennighoff/flan",
                                    streaming=True, split="train")
        self.loader = iter(self.dataset)

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        # イテレータは自分自身を返す
        return self

    def __next__(self):
        d = next(self.loader)
        q = d["inputs"]
        a = d["targets"]
        if q is None or a is None:
            return {"text": ""}
        txt = "Q: "+q+"\n\nA: "+a
        d["text"] = txt
        return d


class AltParallelEnJaDataset:
    def __init__(self, streaming=True,
                 repo_name="hpprc/alt-parallel-en-ja",
                 mode="train",
                 data_files=None,
                 ):
        self.dataset = load_dataset(
            repo_name,
            split=mode,
            data_files=data_files,
            streaming=streaming)
        self.loader = iter(self.dataset)

    def __iter__(self):
        # イテレータは自分自身を返す
        return self

    def __next__(self):
        d = next(self.loader)
        try:
            d["text"] = d["en"]+"\n"+d["ja"]
        except Exception as e:
            print(d, e)
            return {"text": ""}
        return d


class SodaJaDataset:
    def __init__(self, streaming=True,
                 repo_name="atsushi3110/soda-ja-instruction",
                 data_files="https://huggingface.co/datasets/atsushi3110/soda-ja-instruction/resolve/main/train.json",
                 mode="train"):
        self.dataset = load_dataset(
            repo_name,
            split=mode,
            streaming=streaming)
        self.loader = iter(self.dataset)

    def __iter__(self):
        # イテレータは自分自身を返す
        return self

    def __next__(self):
        try:
            d = next(self.loader)
        except:
            return {"text": ""}
        text = d["text"]
        text = text[text.find("# Input")+10:]
        text = text.replace(" ", "")
        lines = text.split("\n")
        lines = [line for line in lines if line[0] != "#"]
        d["text"] = "\n".join(lines)
        return d


class ShosetuSevenK:
    def __init__(self, streaming=True,
                 repo_name="RyokoAI/Syosetu711K",
                 mode="train"):
        self.dataset = load_dataset(
            repo_name,
            split=mode,
            streaming=streaming)
        self.loader = iter(self.dataset)

    def __iter__(self):
        # イテレータは自分自身を返す
        return self

    def __next__(self):
        d = next(self.loader)
        lines = d["text"].split("\n")
        n_lines = lines[0:1]+(lines[7:])
        n_lines = [i for i in n_lines if i != ""]
        n_lines = [i for i in n_lines if i[0] != "【"]
        text = "\n".join(n_lines)
        d["text"] = text
        return d


class GitHubCodePythonDataset:
    def __init__(self, streaming=True,
                 ):
        self.dataset = load_dataset("loubnabnl/github-code-more-filtering",
                                    split="train",
                                    streaming=streaming,
                                    )
        self.dataset = self.dataset.filter(lambda x: x["language"] == "Python")
        self.loader = iter(self.dataset)

    def __iter__(self):
        # イテレータは自分自身を返す
        return self

    def __next__(self):
        d = next(self.loader)
        d["text"] = d["code"]
        return d


class OasstDataset:
    def __init__(self, streaming=True,
                 ):
        self.dataset = load_dataset("sablo/oasst2_curated",
                                    split="train",
                                    streaming=streaming,
                                    )
        self.loader = iter(self.dataset)

    def __iter__(self):
        # イテレータは自分自身を返す
        return self

    def __next__(self):
        d = next(self.loader)
        talk = ""
        for line in d["messages"]:
            talk += line["role"]+": "+line["content"]+"\n"
        d["text"] = talk
        return d


class DollyDataset:
    def __init__(self, streaming=True,
                 ):
        self.dataset = load_dataset("databricks/databricks-dolly-15k",
                                    split="train",
                                    streaming=streaming,
                                    )
        self.loader = iter(self.dataset)

    def __iter__(self):
        # イテレータは自分自身を返す
        return self

    def __next__(self):
        d = next(self.loader)
        talk = ""
        talk += "user: "+d["instruction"]+"\n"
        if "context" in d:
            talk += d["context"]+"\n"
        talk += "assistant: "+d["response"]
        d["text"] = talk
        return d
