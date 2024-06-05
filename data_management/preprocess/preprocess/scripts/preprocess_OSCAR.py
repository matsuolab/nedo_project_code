import json
from multiprocessing import Pool
from tqdm.contrib import tqdm
from preprocess import dedup
from preprocess import filters
from pathlib import Path
from argparse import ArgumentParser


def read_jsonl(path, chunk_size=10000):
    with open(path, 'r') as f:
        chunk = []
        for i, line in enumerate(f):
            line = line.strip()
            if line == '':
                continue
            chunk.append(json.loads(line))
            if i % chunk_size == chunk_size - 1:
                yield chunk
                chunk = []
        yield chunk


def process_json(path_input, path_output, functions=None,
                 chunk_size=10000, num_worker=1, return_dataset=False):
    if return_dataset:
        dataset = []
    with open(path_output, 'w') as f:
        for chunk in read_jsonl(path_input, chunk_size):
            if functions is None:
                continue
            for func_type, func in functions:
                if func_type == 'filter':
                    chunk = list(filter(func, chunk))
                if func_type == 'map':
                    pool = Pool(processes=num_worker)
                    with tqdm(total=len(chunk)) as t:
                        processed = []
                        for data in pool.imap(func, chunk):
                            processed.append(data)
                            t.update(1)
                        chunk = processed
            text = '\n'.join([json.dumps(d, ensure_ascii=False) for d in chunk])
            f.write(text+'\n')
            if return_dataset:
                dataset += chunk
    if return_dataset:
        return dataset


_hasher = dedup.Hasher(10, 200, 20, 10)
def compute_LSH(text):
    _text = dedup.text(text)
    _hasher.apply(_text)
    return _text.getHashes()


domain_checker = filters.has_valid_domain('../dict/ja_valid_domains.txt')
count_NG_word_AD = filters.gen_NG_word_counter('../dict/ja_advertisement_keywords.txt')
count_NG_word_adult = filters.gen_NG_word_counter('../dict/ja_adult_keywords.txt', ignore_confused=True)
count_NG_word_violence = filters.gen_NG_word_counter('../dict/ja_violence_keywords.txt', ignore_confused=True)
count_NG_word_discrimination = filters.gen_NG_word_counter('../dict/ja_discrimination_keywords.txt', ignore_confused=True)


def map_function_mC4(data):
    text = filters.normalize(data['text'])
    meta = data.get('meta', {})
    meta.update({k: v for k, v in data.items() if k not in {'text', 'meta'}})
    # add filters info
    meta['is_not_empty'] = filters.is_not_empty(text)
    meta['is_japanese'] = filters.is_japanese(text)
    meta['has_valid_domain'] = domain_checker(meta['url'])
    meta['average_sentence_length'] = filters.average_sentence_length(text)
    meta['compression_ratio'] = filters.compression_ratio(text)
    meta['count_NG_word_AD'] = count_NG_word_AD(text)
    meta['count_NG_word_adult'] = count_NG_word_adult(text)
    meta['count_NG_word_violence'] = count_NG_word_violence(text)
    meta['count_NG_word_discrimination'] = count_NG_word_discrimination(text)
    data = {'text': text, 'meta': meta}
    return data


def map_function_OSCAR(data):
    # text = filters.normalize(data['content'])
    text = data['text']
    meta = data.get('meta', {})
    # meta.update({k: v for k, v in data.items() if k not in {'content', 'meta'}})
    # meta['is_not_empty'] = filters.is_not_empty(text)
    # meta['is_japanese'] = filters.is_japanese(text)
    # meta['has_valid_domain'] = domain_checker(meta['warc_headers']['warc-target-uri'])
    meta['average_sentence_length'] = filters.average_sentence_length(text)
    # meta['compression_ratio'] = filters.compression_ratio(text)
    # meta['count_NG_word_AD'] = count_NG_word_AD(text)
    # meta['count_NG_word_adult'] = count_NG_word_adult(text)
    # meta['count_NG_word_violence'] = count_NG_word_violence(text)
    # meta['count_NG_word_discrimination'] = count_NG_word_discrimination(text)
    data = {'text': text, 'meta': meta}
    return data


def map_function_dedup(data):
    text = data['text']
    meta = data.get('meta', {})
    meta['dedup_LSHs'] = compute_LSH(text) 
    return data


def filter_function(data):
    meta = data['meta']
    if not meta['is_not_empty']:
        return False
    if not meta['is_japanese']:
        return False
    if not meta['has_valid_domain']:
        return False
    if meta['average_sentence_length'] > 250:
        return False
    if not (0.3 < meta['compression_ratio'] < 1.):
        return False
    if meta['count_NG_word_AD'] > 12:
        return False
    if meta['count_NG_word_adult'] > 5:
        return False
    if meta['count_NG_word_violence'] > 20:
        return False
    if meta['count_NG_word_discrimination'] > 20:
        return False
    return True


def main(base_dir, index_from, index_to, num_worker=8):
    for index in range(index_from, index_to+1):
        path_input = Path(base_dir).joinpath(f'ja_meta_part_{index:03}.jsonl').as_posix()
        path_output = f'/persistentshare/storage/team_haijima/dataset_pre/20240418_preprocess/datasets/OSCAR/ja_meta_part_{index:03}.jsonl'
        process_json(path_input, path_output, 
                     [('map', map_function_OSCAR), ('filter', filter_function), ('map', map_function_dedup)],
                     num_worker=num_worker, chunk_size=20000, return_dataset=False)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        "--index_from",
        type=int,
        help="index_from",
    )
    parser.add_argument(
        "--index_to",
        type=int,
        help="index_to",
    )
    parser.add_argument(
        "--num_worker",
        type=int,
        help="index_to",
    )
    args = parser.parse_args()
    base_dir = '/persistentshare/storage/team_haijima/dataset_pre/20240416_OSCAR_filtering/filtered/'
    main(base_dir, args.index_from, args.index_to, args.num_worker)
