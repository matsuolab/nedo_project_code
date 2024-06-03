import logging
import pathlib
import time
import typing
from argparse import ArgumentParser
from typing import Any, Callable

import tqdm
from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    disable_caching,
    load_dataset,
)
from datasets.splits import Split
from filters import (
    extract_japanese_text,
    has_good_average_sentence_length,
    has_good_compression_ratio,
    has_valid_alphanum_fraction,
    has_valid_avg_line_length,
    is_not_empty_url,
    has_valid_domain,
    is_not_blacklist_domain,
    is_not_additional_blacklist_domain,
    is_japanese_by_fasttext,
    has_valid_extension,
    has_valid_max_line_length,
    is_japanese,
    is_not_ad_content,
    is_not_empty,
    reformat_data,
    remove_empty_parenthesis,
    remove_wikipedia_footnote,
    has_below_duplicate_line_char_ratio,
    has_below_duplicate_line_ratio,
    has_below_duplicate_paragraph_char_ratio,
    has_below_duplicate_paragraph_ratio,
    has_below_max_ngram_ratio,
    has_below_repeated_ngram_ratio,
    has_good_average_sentence_length_by_swallow,
    has_sentence_with_min_length,
    has_documents_with_min_length,
    has_valid_hiragana_fraction,
    has_valid_katakana_fraction,
    mask_phone_and_email,
    remove_urlj,
    remove_strange,
    is_not_adult_content,
    is_not_discrimination_content,
    is_not_violence_content,
    has_valid_ending,
    remove_copyright,
    has_valid_japanesenum_fraction,
)

logger = logging.getLogger(__name__)
disable_caching()

CHUNK_SIZE = 100_000


def get_data_files(search_dir: pathlib.Path, ext: str) -> dict[Split, pathlib.Path]:
    train_files = list(search_dir.glob(f"*train*.{ext}"))
    valid_files = list(search_dir.glob(f"*valid*.{ext}"))
    test_files = list(search_dir.glob(f"*test*.{ext}"))
    assert len(train_files) == 1, f"Found {len(train_files)} train files."
    assert len(valid_files) <= 1, f"Found {len(valid_files)} valid files."
    assert len(test_files) <= 1, f"Found {len(test_files)} test files."
    data_files = {Split.TRAIN: train_files[0]}
    if len(valid_files) == 1:
        data_files[Split.VALIDATION] = valid_files[0]
    if len(test_files) == 1:
        data_files[Split.TEST] = test_files[0]
    return data_files


def reformat_and_filter_dataset(
    dataset: DatasetDict, dataset_name: str, strict: bool = False
) -> DatasetDict:
    reformat_fn: Callable[..., dict[str, Any]]
    map_fns: list[Callable[..., dict[str, Any]]] = []
    filter_fns: list[Callable[..., bool]] = []
    rephrasing_fns: list[Callable[..., bool]] = []
    if dataset_name == "ja_wiki":
        reformat_fn = reformat_data("text")
        map_fns.append(remove_wikipedia_footnote())
        map_fns.append(remove_empty_parenthesis())
        filter_fns.append(is_not_empty())
    elif dataset_name == "en_wiki":
        reformat_fn = reformat_data("text")
        map_fns.append(remove_wikipedia_footnote())
        map_fns.append(remove_empty_parenthesis())
        filter_fns.append(is_not_empty())
    elif dataset_name == "ja_cc":
        reformat_fn = reformat_data("text")
        map_fns.append(extract_japanese_text())
        filter_fns.append(has_valid_domain())
        filter_fns.append(is_not_empty())
        filter_fns.append(is_japanese())
        filter_fns.append(is_not_ad_content())
        max_allowed_num: int = 2 if strict else 3
        filter_fns.append(is_not_adult_content(max_allowed_num))
        filter_fns.append(is_not_discrimination_content(max_allowed_num))
        filter_fns.append(is_not_violence_content(max_allowed_num))
        max_average_sentence_length: int = 80 if strict else 250
        filter_fns.append(has_good_average_sentence_length(max_average_sentence_length))
        min_score = 0.375 if strict else 0.30
        max_score = 0.70
        filter_fns.append(has_good_compression_ratio(min_score, max_score))
    elif dataset_name == "en_pile":
        reformat_fn = reformat_data("text")
        filter_fns.append(is_not_empty())
        filter_fns.append(lambda x: x["meta"]["pile_set_name"] != "Books3")
    elif dataset_name == "code_stack":
        reformat_fn = reformat_data("content")
        filter_fns.append(has_valid_extension())
        filter_fns.append(has_valid_max_line_length())
        filter_fns.append(has_valid_avg_line_length())
        filter_fns.append(has_valid_alphanum_fraction())
        filter_fns.append(is_not_empty())
    elif dataset_name == "test":
        reformat_fn = reformat_data("text")
        filter_fns.append(is_not_empty_url())
        filter_fns.append(has_valid_domain())
        filter_fns.append(is_not_blacklist_domain())
        filter_fns.append(is_not_additional_blacklist_domain())
        filter_fns.append(is_japanese_by_fasttext())
        filter_fns.append(has_below_duplicate_line_ratio())
        filter_fns.append(has_below_duplicate_paragraph_ratio())
        filter_fns.append(has_below_duplicate_line_char_ratio())
        filter_fns.append(has_below_duplicate_paragraph_char_ratio())
        filter_fns.append(has_below_max_ngram_ratio(n=2, max_ratio=0.20))
        filter_fns.append(has_below_max_ngram_ratio(n=3, max_ratio=0.18))
        filter_fns.append(has_below_max_ngram_ratio(n=4, max_ratio=0.16))
        filter_fns.append(has_below_repeated_ngram_ratio(n=5, max_ratio=0.15))
        filter_fns.append(has_below_repeated_ngram_ratio(n=6, max_ratio=0.14))
        filter_fns.append(has_below_repeated_ngram_ratio(n=7, max_ratio=0.13))
        filter_fns.append(has_below_repeated_ngram_ratio(n=8, max_ratio=0.12))
        filter_fns.append(has_below_repeated_ngram_ratio(n=9, max_ratio=0.11))
        filter_fns.append(has_below_repeated_ngram_ratio(n=10, max_ratio=0.10))
        filter_fns.append(has_good_average_sentence_length_by_swallow())
        filter_fns.append(has_sentence_with_min_length())
        filter_fns.append(has_documents_with_min_length())
        filter_fns.append(has_valid_japanesenum_fraction())
        filter_fns.append(has_valid_hiragana_fraction())
        filter_fns.append(has_valid_katakana_fraction())
        filter_fns.append(is_not_ad_content(max_allowed_num=8))
        map_fns.append(mask_phone_and_email())
        map_fns.append(remove_urlj())
        map_fns.append(remove_strange())
        filter_fns.append(is_not_adult_content())
        filter_fns.append(is_not_discrimination_content())
        filter_fns.append(is_not_violence_content())
        map_fns.append(remove_copyright())
        filter_fns.append(has_valid_ending(max_ratio=0.2))
    elif dataset_name == "cc":
        reformat_fn = reformat_data("text")
        # 前処理
        map_fns.append(mask_phone_and_email())
        map_fns.append(remove_urlj())
        map_fns.append(remove_strange())
        map_fns.append(remove_copyright())
        # URLフィルタリング
        filter_fns.append(is_not_empty_url())
        filter_fns.append(has_valid_domain())
        filter_fns.append(is_not_blacklist_domain())
        filter_fns.append(is_not_additional_blacklist_domain())
        filter_fns.append(is_japanese_by_fasttext())
        # ルールベースフィルタリング① アダルト系などの有害コンテンツや情報量が少なすぎる文章の削除
        filter_fns.append(has_below_duplicate_line_ratio())
        filter_fns.append(has_below_duplicate_paragraph_ratio())
        filter_fns.append(has_below_duplicate_line_char_ratio())
        filter_fns.append(has_below_duplicate_paragraph_char_ratio())
        filter_fns.append(has_below_max_ngram_ratio(n=2, max_ratio=0.20))
        filter_fns.append(has_below_max_ngram_ratio(n=3, max_ratio=0.18))
        filter_fns.append(has_below_max_ngram_ratio(n=4, max_ratio=0.16))
        filter_fns.append(has_below_repeated_ngram_ratio(n=5, max_ratio=0.35))
        filter_fns.append(has_below_repeated_ngram_ratio(n=6, max_ratio=0.35))
        filter_fns.append(has_below_repeated_ngram_ratio(n=7, max_ratio=0.30))
        filter_fns.append(has_below_repeated_ngram_ratio(n=8, max_ratio=0.25))
        filter_fns.append(has_below_repeated_ngram_ratio(n=9, max_ratio=0.25))
        filter_fns.append(has_below_repeated_ngram_ratio(n=10, max_ratio=0.10))
        filter_fns.append(has_valid_japanesenum_fraction())
        filter_fns.append(has_valid_hiragana_fraction(allowed_hiragana_fraction=0.10))
        filter_fns.append(has_valid_katakana_fraction())
        filter_fns.append(is_not_adult_content())
        filter_fns.append(is_not_discrimination_content())
        filter_fns.append(is_not_violence_content())
        filter_fns.append(has_valid_ending(max_ratio=0.2))
        # ルールベースフィルタリング② キーワードの羅列などで内容は有益だが形式が悪い文章をRepharasingへ回す
        rephrasing_fns.append(has_good_average_sentence_length_by_swallow())
        rephrasing_fns.append(has_sentence_with_min_length())
        rephrasing_fns.append(has_documents_with_min_length())
        rephrasing_fns.append(is_not_ad_content(max_allowed_num=8))
    elif dataset_name == "cuX":
        reformat_fn = reformat_data("text")
        # 前処理
        map_fns.append(mask_phone_and_email())
        map_fns.append(remove_urlj())
        map_fns.append(remove_strange())
        map_fns.append(remove_copyright())
        # URLフィルタリング
        filter_fns.append(is_not_blacklist_domain())
        filter_fns.append(is_not_additional_blacklist_domain())
        filter_fns.append(is_japanese_by_fasttext())
        # ルールベースフィルタリング① アダルト系などの有害コンテンツや情報量が少なすぎる文章の削除
        filter_fns.append(has_below_duplicate_line_ratio())
        filter_fns.append(has_below_duplicate_paragraph_ratio())
        filter_fns.append(has_below_duplicate_line_char_ratio())
        filter_fns.append(has_below_duplicate_paragraph_char_ratio())
        filter_fns.append(has_below_max_ngram_ratio(n=2, max_ratio=0.20))
        filter_fns.append(has_below_max_ngram_ratio(n=3, max_ratio=0.18))
        filter_fns.append(has_below_max_ngram_ratio(n=4, max_ratio=0.16))
        filter_fns.append(has_below_repeated_ngram_ratio(n=5, max_ratio=0.35))
        filter_fns.append(has_below_repeated_ngram_ratio(n=6, max_ratio=0.35))
        filter_fns.append(has_below_repeated_ngram_ratio(n=7, max_ratio=0.30))
        filter_fns.append(has_below_repeated_ngram_ratio(n=8, max_ratio=0.25))
        filter_fns.append(has_below_repeated_ngram_ratio(n=9, max_ratio=0.25))
        filter_fns.append(has_below_repeated_ngram_ratio(n=10, max_ratio=0.10))
        filter_fns.append(has_valid_japanesenum_fraction())
        filter_fns.append(has_valid_hiragana_fraction(allowed_hiragana_fraction=0.10))
        filter_fns.append(has_valid_katakana_fraction())
        filter_fns.append(is_not_adult_content())
        filter_fns.append(is_not_discrimination_content())
        filter_fns.append(is_not_violence_content())
        filter_fns.append(has_valid_ending(max_ratio=0.2))
        # ルールベースフィルタリング② キーワードの羅列などで内容は有益だが形式が悪い文章をRepharasingへ回す
        rephrasing_fns.append(has_good_average_sentence_length_by_swallow())
        rephrasing_fns.append(has_sentence_with_min_length())
        rephrasing_fns.append(has_documents_with_min_length())
        rephrasing_fns.append(is_not_ad_content(max_allowed_num=8))
    elif dataset_name == "ja2010":
        reformat_fn = reformat_data("text")
        # 前処理
        map_fns.append(mask_phone_and_email())
        map_fns.append(remove_urlj())
        map_fns.append(remove_strange())
        map_fns.append(remove_copyright())
        # URLフィルタリング
        filter_fns.append(is_japanese_by_fasttext())
        # ルールベースフィルタリング① アダルト系などの有害コンテンツや情報量が少なすぎる文章の削除
        filter_fns.append(has_below_duplicate_line_ratio(max_duplicate_ratio=0.65))
        filter_fns.append(has_below_duplicate_paragraph_ratio())
        filter_fns.append(has_below_duplicate_line_char_ratio(max_duplicate_char_ratio=0.55))
        filter_fns.append(has_below_duplicate_paragraph_char_ratio())
        filter_fns.append(has_below_max_ngram_ratio(n=2, max_ratio=0.03))
        filter_fns.append(has_below_max_ngram_ratio(n=3, max_ratio=0.025))
        filter_fns.append(has_below_max_ngram_ratio(n=4, max_ratio=0.02))
        filter_fns.append(has_below_repeated_ngram_ratio(n=5, max_ratio=0.75))
        filter_fns.append(has_below_repeated_ngram_ratio(n=6, max_ratio=0.65))
        filter_fns.append(has_below_repeated_ngram_ratio(n=7, max_ratio=0.65))
        filter_fns.append(has_below_repeated_ngram_ratio(n=8, max_ratio=0.65))
        filter_fns.append(has_below_repeated_ngram_ratio(n=9, max_ratio=0.65))
        filter_fns.append(has_below_repeated_ngram_ratio(n=10, max_ratio=0.65))
        filter_fns.append(has_valid_japanesenum_fraction())
        filter_fns.append(has_valid_hiragana_fraction(allowed_hiragana_fraction=0.10))
        filter_fns.append(has_valid_katakana_fraction())
        filter_fns.append(is_not_adult_content(max_allowed_num=150))
        filter_fns.append(is_not_discrimination_content(max_allowed_num=30))
        filter_fns.append(is_not_violence_content(max_allowed_num=25))
        filter_fns.append(has_valid_ending(max_ratio=0.2))
        # ルールベースフィルタリング② キーワードの羅列などで内容は有益だが形式が悪い文章をRepharasingへ回す
        rephrasing_fns.append(has_good_average_sentence_length_by_swallow())
        rephrasing_fns.append(has_documents_with_min_length())
        rephrasing_fns.append(is_not_ad_content(max_allowed_num=30))
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}.")

    dataset = dataset.map(reformat_fn, batched=False)
    train_dataset: typing.Union[Dataset, IterableDataset] = dataset["train"]
    if isinstance(train_dataset, Dataset):
        columns = list(train_dataset[0].keys())
    elif isinstance(train_dataset, IterableDataset):
        columns = list(train_dataset.take(1))[0].keys()
    else:
        raise ValueError
    dataset = dataset.map(remove_columns=list(set(columns) - {"text", "meta"}))
    for map_fn in map_fns:
        dataset = dataset.map(map_fn, batched=False)
    for filter_fn in filter_fns:
        dataset = dataset.filter(filter_fn)

    def apply_rephrasing_fns(element: dict[str, Any]) -> dict[str, bool]:
        for rephrasing_fn in rephrasing_fns:
            if not rephrasing_fn(element):
                return {"rephrasing": True}
        return {"rephrasing": False}

    dataset = dataset.map(apply_rephrasing_fns, batched=False)
    return dataset.filter(is_not_empty())


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "DATASET_NAME",
        type=str,
        choices=[
            "ja_wiki",
            "en_wiki",
            "ja_cc",
            "en_pile",
            "code_stack",
            "test",
            "cc",
            "cuX",
            "ja2010"
        ],
        help="Dataset name",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        help="Path to the data directory.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Path to the output directory.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Whether to use strict filtering.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Whether to overwrite the output directory.",
    )
    args = parser.parse_args()

    input_dir: pathlib.Path = pathlib.Path(args.input_dir)
    output_dir: pathlib.Path = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    logger.info("Loading the dataset")
    dataset: DatasetDict = load_dataset(
        "json",
        data_files={k: str(v) for k, v in get_data_files(input_dir, "jsonl").items()},
        streaming=True,
    )

    dataset = reformat_and_filter_dataset(
        dataset, args.DATASET_NAME, strict=args.strict
    )

    logger.info(f"Writing the reformatted data to {output_dir}.")
    for split, ds in dataset.items():
        chunk_index = 0
        for batch in tqdm.tqdm(ds.iter(batch_size=CHUNK_SIZE)):
            output_file: pathlib.Path = output_dir.joinpath(
                f"{split}_{chunk_index}.parquet"
            )
            if output_file.exists() and not args.overwrite:
                logger.error(
                    f"{output_file} already exists. Specify --overwrite to overwrite."
                )
                chunk_index += 1
                continue

            Dataset.from_dict(batch).to_parquet(output_file)
            chunk_index += 1

    end_time = time.time()
    logger.info(
        f"Finished processing the dataset. Elapsed time: \
            {end_time - start_time} [sec]"
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(name)s:%(lineno)d: %(levelname)s: %(message)s",
    )
    main()
