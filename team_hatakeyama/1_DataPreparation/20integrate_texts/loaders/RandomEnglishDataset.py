from datasets import load_dataset
from .CleanedJapaneseWikiDataset import CleanedEngWikiDataset
from .OtherDatasets import *
from .PilePythonDataset import PilePythonDataset


class RandomEnglishDataset:
    def __init__(self, ):
        print("loading many datasets. This may take a while...")
        dataset_list = [

            # pes2o: : 67.56M docs, ca. 47 b tokens
            load_dataset("allenai/peS2o", streaming=True, split="train"),

            # wikipedia: 6458000 docs
            CleanedEngWikiDataset(),

            # wikibook: 50k docs
            WikiBookEn(),

            # stack exchange 60k
            PileStackExchange(),
            PileStackExchange(mode="test"),

            # python code 49k
            PythonCodeDataset(),

            # openmathinstruct ja: 1820000 docs
            OpenMathInstructJa(),

            # the pile python dataset
            # 6,098.8 M ?
            PilePythonDataset("../data/original_dump/python/"),

            # openmath 6.3 million
            load_dataset("open-web-math/open-web-math",
                         streaming=True, split="train"),

            # flan:;  3,530,340
            FlanDataset(),

            # 日英: 20k
            AltParallelEnJaDataset(),

            # python code: 6 million
            GitHubCodePythonDataset(),

            # oasst 5k
            OasstDataset(),

            # dolly 15k
            DollyDataset(),

            # cosmopedia
            load_dataset("HuggingFaceTB/cosmopedia",
                         "stanford", split="train", streaming=True,),
            load_dataset("HuggingFaceTB/cosmopedia",
                         "wikihow", split="train", streaming=True,),
            load_dataset("HuggingFaceTB/cosmopedia",
                         "openstax", split="train", streaming=True,),
            load_dataset("HuggingFaceTB/cosmopedia",
                         "khanacademy", split="train", streaming=True,),
            load_dataset("HuggingFaceTB/cosmopedia",
                         "auto_math_text", split="train", streaming=True,),

            # PMC 3k
            PMCDataset(),
        ]

        self.dataset_list = dataset_list
        self.loader_list = [iter(d) for d in self.dataset_list]

    def __iter__(self):
        # イテレータは自分自身を返す
        return self

    def __next__(self):
        if len(self.loader_list) == 0:
            raise StopIteration

        loader = random.choice(self.loader_list)

        try:
            d = next(loader)
            return d
        except StopIteration:
            self.loader_list.remove(loader)
            return {"text": ""}
