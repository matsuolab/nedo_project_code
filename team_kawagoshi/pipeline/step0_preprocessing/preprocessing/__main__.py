import os
import re
import json
import difflib
import MeCab
import argparse
import unicodedata
import multiprocessing
import subprocess
from abc import ABC, abstractmethod
from tqdm import tqdm
from collections import Counter
from hojichar import document_filters, Compose, Document

def parse_args():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Preprocess dataset')
    parser.add_argument('--input_file', type=str, required=True, help='Input file path')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory path')
    parser.add_argument('--min_doc_len', type=int, default=50, help='Minimum document length')
    parser.add_argument('--max_doc_len', type=int, default=50000, help='Maximum document length')
    parser.add_argument('--symbol_noun_ratio', type=float, default=0.7, help='Maximum ratio of symbols and nouns')
    parser.add_argument('--unwanted_strings_file', type=str, default='unwanted_strings.json', help='JSON file containing unwanted strings')
    parser.add_argument('--workers', type=int, default=multiprocessing.cpu_count(), help='Number of worker processes')
    return parser.parse_args()


class Processer(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def execute(self, data):
        self.data = data
        pass


class DatasetPreprocessor(Processer):
    def __init__(self, args):
        self.args = args

    @property
    def tagger(self):
        return MeCab.Tagger("")

    def _load_unwanted_strings(self, file_path) -> dict:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)

    def _process_line(self, line, unwanted_strings_dict):
        # Load the JSON data from the line
        data = json.loads(line)
        paragraph = data['text']

        # Process the paragraph
        processed_paragraph = ParagraphProcesser(paragraph, self.args.min_doc_len, self.args.max_doc_len).execute()

        if processed_paragraph is None:
            return None

        # Process sentences in the paragraph
        processed_sentences = SentenceProcesser(processed_paragraph, self.args.symbol_noun_ratio, unwanted_strings_dict, self.tagger).execute()

        # Return the processed sentences if not empty
        if processed_sentences:
            data['text'] = ''.join(processed_sentences)
            return json.dumps(data, ensure_ascii=False)
        else:
            return None

    def process_dataset(self):
        # Load unwanted strings from the specified file
        unwanted_strings_dict = self._load_unwanted_strings(self.args.unwanted_strings_file)

        # Get the input file prefix and construct output file paths
        input_file_prefix_path = os.path.splitext(os.path.basename(self.args.input_file))[0]
        output_processed_file_path = os.path.join(self.args.output_dir, f"{input_file_prefix_path}_processed.jsonl")
        output_rejected_file_path = os.path.join(self.args.output_dir, f"{input_file_prefix_path}_rejected.jsonl")

        # Load the progress from the progress file
        progress = self._load_progress(self.args.output_dir, input_file_prefix_path)

        # Open the input file, output file, and rejected file
        with open(self.args.input_file, 'r', encoding='utf-8') as infile, \
            open(output_processed_file_path, 'a' if progress else 'w', encoding='utf-8') as outfile, \
            open(output_rejected_file_path, 'a' if progress else 'w', encoding='utf-8') as rejected_outfile:

            # Create a process pool with the specified number of worker processes
            pool = multiprocessing.Pool(processes=self.args.workers)

            # Iterate over the lines in the input file with progress tracking
            for line_num, line in enumerate(tqdm(infile, desc="Processing lines", initial=progress, total=progress+1)):
                # Skip lines that have already been processed
                if line_num < progress:
                    continue

                # Process the line using the process pool
                result = pool.apply_async(self._process_line, args=(line, unwanted_strings_dict))

                # Write the processed line to the output file or rejected file
                processed_line = result.get()
                if processed_line is not None:
                    outfile.write(processed_line + '\n')
                else:
                    rejected_outfile.write(line)

                # Save progress every 1000 lines
                if line_num % 1000 == 0:
                    self._save_progress(self.args.output_dir, input_file_prefix_path, line_num + 1)

            # Close the process pool
            pool.close()
            pool.join()

        # Save the final progress
        self._save_progress(self.args.output_dir, input_file_prefix_path, line_num + 1)

    def _get_progress_file_path(self, output_dir, input_file_prefix):
        # Get the path of the progress file
        return os.path.join(output_dir, f"{input_file_prefix}_progress.txt")

    def _save_progress(self, output_dir, input_file_prefix, line_num):
        # Save the progress to a temporary file and replace the progress file
        progress_file_path = self._get_progress_file_path(output_dir, input_file_prefix)
        temp_file_path = progress_file_path + '.temp'

        with open(temp_file_path, 'w', encoding='utf-8') as temp_file:
            temp_file.write(str(line_num))

        os.replace(temp_file_path, progress_file_path)

    def _load_progress(self, output_dir, input_file_prefix):
        # Load the progress from the progress file if it exists
        progress_file_path = self._get_progress_file_path(output_dir, input_file_prefix)
        if os.path.exists(progress_file_path):
            with open(progress_file_path, 'r', encoding='utf-8') as file:
                return int(file.read().strip())
        return 0

    def execute(self):
        self.process_dataset()


class ParagraphProcesser(Processer):
    def __init__(self, data, min_doc_len, max_doc_len):
        self.data = data
        self.min_doc_len = min_doc_len
        self.max_doc_len = max_doc_len
        return None
    
    def filtering_length_contents(self):
        """① 長さフィルタと除去"""
        if len(self.data) < self.min_doc_len or len(self.data) > self.max_doc_len:
            return None
        return self.data

    def exec_normalize_contents(self):
        """② 基本正規化の実行"""
        self.data = TextNormalizer.normalize_neologd(self.data)
        return self.data

    def execute(self):
        self.data = self.filtering_length_contents()
        if self.data is None:
            return None
        self.data = self.exec_normalize_contents()
        return self.data
    

class SentenceProcesser(Processer):
    def __init__(self, data, symbol_noun_ratio, unwanted_strings, tagger):
        self.data = data
        self.symbol_noun_ratio = symbol_noun_ratio
        self.unwanted_strings = unwanted_strings
        self.tagger = tagger

    def _delete_short_sentence(self, sentence):
        """③ 1文が10文字以下の文を削除"""
        if len(sentence) <= 10:
            return None
        return sentence
    
    def _delete_high_noun_rate_sentence(self, sentence):
        """⑤ 記号・名詞割合が高い文章の除去"""
        return TextFilter.filter_line(sentence, self.tagger, self.symbol_noun_ratio)
    
    def _delete_NG_sentence(self, sentence):
        """④ NGコンテンツの除去"""
        cleaner = Compose([
            document_filters.DiscardAdultContentJa(),
            document_filters.DiscardAds(),
            document_filters.DiscardViolenceContentJa(),
            document_filters.DiscardDiscriminationContentJa(),
            document_filters.MaskPersonalInformation()
        ])
        return cleaner.apply(Document(sentence)).text

    def _delete_other_stranges(self, sentence):
        """⑥ その他消したい文字列の削除"""
        for unwanted_string in self.unwanted_strings["unwanted_word"]:
            if unwanted_string in sentence:
                return None
        return sentence

    def execute(self):
        processed_sentences = []
        
        non_english_chars = r'[^a-zA-Z]'
        self.data = re.sub(f'({non_english_chars})\n({non_english_chars})', r'\1。\2', self.data)

        sentences = re.findall(r'[^。！？]*[。！？]', self.data)
            
        for sentence in sentences:
            sentence = sentence.strip()
            sentence = self._delete_short_sentence(sentence) 
            if sentence is None:
                continue
            sentence = self._delete_high_noun_rate_sentence(sentence)
            if sentence is None:
                continue
            sentence = self._delete_NG_sentence(sentence)
            sentence = self._delete_other_stranges(sentence)
            if sentence is None:
                continue
            processed_sentences.append(sentence)

        return processed_sentences


class TextNormalizer:
    @staticmethod
    def unicode_normalize(self, s):
        pt = re.compile('([{}]+)'.format(self))
        def norm(c):
            return unicodedata.normalize('NFKC', c) if pt.match(c) else c
        s = ''.join(norm(x) for x in re.split(pt, s))
        s = re.sub('－', '-', s)
        return s

    @staticmethod
    def remove_extra_spaces(s):
        s = re.sub('[ 　]+', ' ', s)
        blocks = ''.join(('\u4E00-\u9FFF',  # CJK UNIFIED IDEOGRAPHS
                        '\u3040-\u309F',  # HIRAGANA
                        '\u30A0-\u30FF',  # KATAKANA
                        '\u3000-\u303F',  # CJK SYMBOLS AND PUNCTUATION
                        '\uFF00-\uFFEF'   # HALFWIDTH AND FULLWIDTH FORMS
                        ))
        basic_latin = '\u0000-\u007F'

        def remove_space_between(cls1, cls2, s):
            p = re.compile('([{}]) ([{}])'.format(cls1, cls2))
            while p.search(s):
                s = p.sub(r'\1\2', s)
            return s

        s = remove_space_between(blocks, blocks, s)
        s = remove_space_between(blocks, basic_latin, s)
        s = remove_space_between(basic_latin, blocks, s)
        return s

    @staticmethod
    def normalize_neologd(s):
        s = s.strip()
        s = TextNormalizer.unicode_normalize('０-９Ａ-Ｚａ-ｚ｡-ﾟ', s)

        def maketrans(f, t):
            return {ord(x): ord(y) for x, y in zip(f, t)}

        s = re.sub('[˗֊‐‑‒–⁃⁻₋−]+', '-', s)  # normalize hyphens
        s = re.sub('[﹣－ｰ—―─━ー]+', 'ー', s)  # normalize choonpus
        s = re.sub('[~∼∾〜〰～]', '', s)  # remove tildes
        s = s.translate(
            maketrans('!"#$%&\'()*+,-./:;<=>?@[¥]^_`{|}~｡､･｢｣',
                '！”＃＄％＆’（）＊＋，－．／：；＜＝＞？＠［￥］＾＿｀｛｜｝〜。、・「」'))

        s = TextNormalizer.remove_extra_spaces(s)
        s = TextNormalizer.unicode_normalize('！”＃＄％＆’（）＊＋，－．／：；＜＞？＠［￥］＾＿｀｛｜｝〜', s)  # keep ＝,・,「,」
        s = re.sub('[’]', '\'', s)
        s = re.sub('[”]', '"', s)
        # Replace consecutive "w" characters with "。"
        s = re.sub(r'w{2,}', '。', s)
        # remove header-like
        s = re.sub(r'【[^】]*】', '', s)

        return s

    @staticmethod
    def normalize_file(input_file_path, output_file_path):
        with open(input_file_path, 'r', encoding='utf-8') as input_file, \
             open(output_file_path, 'w', encoding='utf-8') as output_file:
            for line in input_file:
                normalized_line = TextNormalizer.normalize_neologd(line)
                output_file.write(normalized_line + '\n')

class TextCleaner:
    @staticmethod
    def char_is_hiragana(c):
        return u'\u3040' <= c <= u'\u309F'

    @staticmethod
    def contains_hiragana(s):
        return any(TextCleaner.char_is_hiragana(c) for c in s)

    @staticmethod
    def count_whitespaces(text):
        return text.count(" ")

    @staticmethod
    def is_closing_brace(c):
        return c in ["」", "》", "）", "〉", "】", "]"]

    @staticmethod
    def do_clean(text: str, ws_threshold=1):
        if not TextCleaner.contains_hiragana(text):
            return None

        non_english_chars = r'[^a-zA-Z]'
        text = re.sub(f'({non_english_chars})\n({non_english_chars})', r'\1。\2', text)
    
        sentences = re.findall(r'[^。！？]*[。！？]', text)

        results = []
        for sent in sentences:
            sent = sent.strip()
            if not sent or TextCleaner.count_whitespaces(sent) >= ws_threshold:
                continue
            results.append(sent)
        return "".join(results) if results else None

class TextFilter:
    @staticmethod
    def filter_line(line, tagger, threshold=0.6):
        if not line.strip():  # check empty lines
            return None
        
        parsed = tagger.parse(line)
        pos_counter = Counter()
        all_counts = 0
        
        for result in parsed.split('\n'):
            if result == 'EOS' or not result:
                continue
            parts = result.split('\t')
            if len(parts) < 2:
                continue
            pos_info = parts[1].split(',')[0]
            pos_counter[pos_info] += 1
            all_counts += 1
        
        if all_counts == 0:  # Prevent division by zero
            return None
        
        meishi_and_symbol_counts = pos_counter['名詞'] + pos_counter.get('記号', 0)
        ratio = meishi_and_symbol_counts / all_counts
        
        if ratio > threshold:
            return None  # If threshold is exceeded, exclude this row
        else:
            return line  # If below threshold, keep this line


if __name__ == '__main__':
    # Parse command-line arguments
    args = parse_args()
    
    # Create an instance of DatasetPreprocessor and process the dataset
    preprocessor = DatasetPreprocessor(args)
    preprocessor.process_dataset()