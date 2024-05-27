import os
import random
from sklearn.model_selection import train_test_split
import MeCab
import fasttext
tagger = MeCab.Tagger()


def wakati_sentence(text):
    words = []
    for c in tagger.parse(text).splitlines()[:-1]:
        try:
            surface, feature = c.split('\t')
        except:
            continue
        surface = feature.split(',')[6]  # 原型に直す
        words.append(surface)
    words = [w for w in words if w != '*']
    return ' '.join(words)


def clean_func(text):
    return text


class TextClassifier:
    def __init__(self,
                 out_path="annotations",
                 max_length=300,
                 ) -> None:
        self.max_length = max_length
        self.model = None

        if not os.path.exists(out_path):
            os.makedirs(out_path)

        self.out_path = out_path
        self.fasttext_path = self.out_path+"/text_labels"

        try:
            self.load_annotations()
        except Exception as e:
            self.good = []
            self.bad = []
            self.annotated = []
            print(e)

    def load_annotations(self):
        with open(f"{self.out_path}/good.txt", "r") as f:
            self.good_texts = f.readlines()
        with open(f"{self.out_path}/bad.txt", "r") as f:
            self.bad_texts = f.readlines()
        self.annotated = self.good_texts + self.bad_texts

    def get_good_annotations(self):
        good_texts = self.good_texts
        good_texts = [f'__label__0 {wakati_sentence(t)}' for t in good_texts]
        return good_texts

    def get_bad_annotations(self):
        bad_texts = self.bad_texts
        bad_texts = [f'__label__1 {wakati_sentence(t)}' for t in bad_texts]
        return bad_texts

    def get_annotated_texts(self, shuffle=True):
        l = self.get_good_annotations() + self.get_bad_annotations()
        if shuffle:
            random.shuffle(l)

        return l

    def wakati_sentence(self, text):
        return wakati_sentence(text)

    def output_annotations(self):
        annotations = self.get_annotated_texts()
        X_train, X_test = train_test_split(
            annotations, test_size=0.2, random_state=100)
        X_valid, X_test = train_test_split(
            X_test, test_size=0.5, random_state=100)

        out_path = self.out_path+"/text_labels"
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        with open(f'{out_path}/fasttext_train.txt', 'w') as temp_file:
            for text in X_train:
                temp_file.write(f"{text}\n")

        with open(f'{out_path}/fasttext_valid.txt', 'w') as temp_file:
            for text in X_valid:
                temp_file.write(f"{text}\n")
        with open(f'{out_path}/fasttext_test.txt', 'w') as temp_file:
            for text in X_test:
                temp_file.write(f"{text}\n")

        print(f"Annotations saved to {out_path}")

        self.fasttext_path = out_path

    def train_fasttext(self,
                       autotuneDuration=600,
                       # epoch=20,
                       wordNgrams=2,
                       auto=True,
                       ):
        print("Outputting annotations")
        self.output_annotations()
        print("Training fasttext model")
        if auto:
            model = fasttext.train_supervised(
                input=f"{self.fasttext_path}/fasttext_train.txt",
                autotuneValidationFile=f"{self.fasttext_path}/fasttext_valid.txt",
                verbose=2
            )
        else:
            model = fasttext.train_supervised(
                input=f"{self.fasttext_path}/fasttext_train.txt",
                autotuneValidationFile=f"{self.fasttext_path}/fasttext_valid.txt",
                autotuneDuration=autotuneDuration,
                wordNgrams=wordNgrams,
                verbose=2
            )

        model.save_model(f"{self.fasttext_path}/model.bin")
        print(f"Model saved to {self.fasttext_path}/model.bin")
        print("Testing model")
        model = fasttext.load_model(f"{self.fasttext_path}/model.bin")
        ret = model.test(f"{self.fasttext_path}/fasttext_test.txt")
        print(ret)

        self.model = model

    def predict(self, text, return_raw=False):
        if self.model is None:
            print("Model is not trained. loading model")
            # load model
            self.model = fasttext.load_model(f"{self.fasttext_path}/model.bin")

        text = self.wakati_sentence(text)
        ret = self.model.predict(text)

        if return_raw:
            return ret
        else:
            return int(ret[0][0].split("_")[-1])

    def clean(self, text, threshold=0.9):
        if text == "":
            return ""
        if text is None:
            return ""
        try:
            pred = self.predict(text, return_raw=True)[1][0]
            # print(pred)
            if pred > threshold:
                return ""
            else:
                return text
        except Exception as e:
            print(e)
            return ""
        if pred == 1:
            return text
        else:
            return ""
