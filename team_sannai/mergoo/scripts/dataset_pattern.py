def get_dataset_pattern(pattern_name):
    if pattern_name == "pattern1":
        ratios = {
            "wiki_ja": 1,
            "dentaku": 1,
            "aozora": 1,
            "basic_math_dentaku": 0.01
        }
        target_list = {
                "wiki_ja": "/storage6/dataset/pretrain/gen_experet/WIKI/raw/wikipedia_ja/merged_wikipedia_ja_16.0.jsonl",
                "dentaku": "/storage6/dataset/pretrain/gen_experet/dentaku/train_data_dentaku_2keta_only_add_1_3keta.jsonl",
                "aozora": "/storage6/corpus/category/BOOK/raw/JA/aozora/ja_book.jsonl",
                "basic_math_dentaku": "/storage6/fujisawa/add_ja_3x3.jsonl",
                # "mc4": "/storage6/dataset/pretrain/router/1B/ja_mc4/merged_mc4_6.0.jsonl"
        }
        return ratios, target_list
    
    if pattern_name == "pattern2":
        ratios = {
           "wiki_ja": 1,
           "dentaku": 1,
           "aozora": 1,
           "basic_math_dentaku": 0.01,
           "wiki_en": 0.4,
        }
        target_list = {
            "wiki_ja": "/storage6/dataset/pretrain/gen_experet/WIKI/raw/wikipedia_ja/merged_wikipedia_ja_16.0.jsonl",
            "dentaku": "/storage6/dataset/pretrain/gen_experet/dentaku/train_data_dentaku_2keta_only_add_1_3keta.jsonl",
            "aozora": "/storage6/corpus/category/BOOK/raw/JA/aozora/ja_book.jsonl",
            "basic_math_dentaku": "/storage6/fujisawa/add_ja_3x3.jsonl",
            "wiki_en": "/storage6/dataset/pretrain/gen_experet/WIKI/raw/wikipedia_en/merged_expert_en_wikipedia_4.0.jsonl"
        }
        return ratios, target_list
    
    if pattern_name == "zoo":
        ratios = {
            "wiki_ja": 0.05,
            "aozora": 0.05,
            "wiki_en": 0.02,
        }
        target_list = {
            "wiki_ja": "/storage6/dataset/pretrain/gen_experet/WIKI/raw/wikipedia_ja/merged_wikipedia_ja_16.0.jsonl",
            "aozora": "/storage6/corpus/category/BOOK/raw/JA/aozora/ja_book.jsonl",
            "wiki_en": "/storage6/dataset/pretrain/gen_experet/WIKI/raw/wikipedia_en/merged_expert_en_wikipedia_4.0.jsonl"
        }
        return ratios, target_list

    if pattern_name == "pattern3":
        ratios = {
           "wiki_ja": 0.13,
           "dentaku": 2.0,
           "basic_math_dentaku": 0.035,
        }
        target_list = {
            "wiki_ja": "/storage6/dataset/pretrain/gen_experet/WIKI/raw/wikipedia_ja/merged_wikipedia_ja_16.0.jsonl",
            "dentaku": "/storage6/dataset/pretrain/gen_experet/dentaku/train_data_dentaku_2keta_only_add_1_3keta.jsonl",
            "basic_math_dentaku": "/storage6/fujisawa/add_ja_3x3_delete_5_pattern.jsonl",
        }
        return ratios, target_list
    
    if pattern_name == "pattern4":
        ratios = {
           "wiki_ja": 0.15,
           "dentaku": 1.0,
           "basic_math_dentaku": 0.0175,
        }
        target_list = {
            "wiki_ja": "/storage6/dataset/pretrain/gen_experet/WIKI/raw/wikipedia_ja/merged_wikipedia_ja_16.0.jsonl",
            "dentaku": "/storage6/dataset/pretrain/gen_experet/dentaku/train_data_dentaku_2keta_only_add_1_3keta.jsonl",
            "basic_math_dentaku": "/storage6/fujisawa/add_ja_3x3_delete_5_pattern.jsonl",
        }
        return ratios, target_list
    
    if pattern_name == "pattern5":
        ratios = {
           "wiki_ja": 0.133,
           "dentaku": 4.80,
        }
        target_list = {
            "wiki_ja": "/storage6/dataset/pretrain/gen_experet/WIKI/raw/wikipedia_ja/merged_wikipedia_ja_16.0.jsonl",
            "dentaku": "/storage6/dataset/pretrain/gen_experet/dentaku/train_data_dentaku_2keta_only_add_1_3keta.jsonl",
        }
        return ratios, target_list

    if pattern_name == "pattern6":
        ratios = {
           "wiki_ja": 0.01,
           "basic_math_dentaku": 0.01,
        }
        target_list = {
            "wiki_ja": "/storage6/dataset/pretrain/gen_experet/WIKI/raw/wikipedia_ja/merged_wikipedia_ja_16.0.jsonl",
            "basic_math_dentaku": "/storage6/fujisawa/add_ja_3x3_delete_5_pattern.jsonl",
        }
        return ratios, target_list

    ## 5/22
    if pattern_name == "pattern7":
        ratios = {
           "wiki_ja": 0.1,
           "dentaku": 2.0,
           "basic_math_dentaku": 0.12,
        }
        target_list = {
            "wiki_ja": "/storage6/dataset/pretrain/gen_experet/WIKI/raw/wikipedia_ja/merged_wikipedia_ja_16.0.jsonl",
            "dentaku": "/storage6/dataset/pretrain/gen_experet/dentaku/train_data_dentaku_2keta_only_add_1_3keta.jsonl",
            "basic_math_dentaku": "/storage6/fujisawa/add_ja_3x3_delete_5_pattern.jsonl",
        }
        return ratios, target_list
    
    if pattern_name == "pattern8":
        ratios = {
           "wiki_ja": 0.067,
           "dentaku": 2.0,
           "basic_math_dentaku": 0.18,
        }
        target_list = {
            "wiki_ja": "/storage6/dataset/pretrain/gen_experet/WIKI/raw/wikipedia_ja/merged_wikipedia_ja_16.0.jsonl",
            "dentaku": "/storage6/dataset/pretrain/gen_experet/dentaku/train_data_dentaku_2keta_only_add_1_3keta.jsonl",
            "basic_math_dentaku": "/storage6/fujisawa/add_ja_3x3_delete_5_pattern.jsonl",
        }
        return ratios, target_list
    
    if pattern_name == "pattern9":
        ratios = {
           "wiki_ja": 0.033,
           "dentaku": 2.00,
           "basic_math_dentaku": 0.24,
        }
        target_list = {
            "wiki_ja": "/storage6/dataset/pretrain/gen_experet/WIKI/raw/wikipedia_ja/merged_wikipedia_ja_16.0.jsonl",
            "dentaku": "/storage6/dataset/pretrain/gen_experet/dentaku/train_data_dentaku_2keta_only_add_1_3keta.jsonl",
            "basic_math_dentaku": "/storage6/fujisawa/add_ja_3x3_delete_5_pattern.jsonl",
        }
        return ratios, target_list
    
    if pattern_name == "pattern10":
        ratios = {
           "wiki_ja": 0.1,
           "dentaku": 10.0,
           "basic_math_dentaku": 0.12,
        }
        target_list = {
            "wiki_ja": "/storage6/dataset/pretrain/gen_experet/WIKI/raw/wikipedia_ja/merged_wikipedia_ja_16.0.jsonl",
            "dentaku": "/storage6/dataset/pretrain/gen_experet/dentaku/train_data_dentaku_2keta_only_add_1_3keta.jsonl",
            "basic_math_dentaku": "/storage6/fujisawa/add_ja_3x3_delete_5_pattern.jsonl",
        }
        return ratios, target_list

    if pattern_name == "pattern11":
        ratios = {
           "wiki_ja": 0.066,
           "dentaku": 20.00,
           "basic_math_dentaku": 0.05,
        }
        target_list = {
            "wiki_ja": "/storage6/dataset/pretrain/gen_experet/WIKI/raw/wikipedia_ja/merged_wikipedia_ja_16.0.jsonl",
            "dentaku": "/storage6/dataset/pretrain/gen_experet/dentaku/train_data_dentaku_2keta_only_add_1_3keta.jsonl",
            "basic_math_dentaku": "/storage6/fujisawa/add_ja_3x3_delete_5_pattern.jsonl",
        }
        return ratios, target_list

    if pattern_name == "pattern12":
        ratios = {
           "wiki_ja": 0.033,
           "dentaku": 2.00,
           "basic_math_dentaku": 0.180,
            "aozora": 0.3,
        }
        target_list = {
            "wiki_ja": "/storage6/dataset/pretrain/gen_experet/WIKI/raw/wikipedia_ja/merged_wikipedia_ja_16.0.jsonl",
            "dentaku": "/storage6/dataset/pretrain/gen_experet/dentaku/train_data_dentaku_2keta_only_add_1_3keta.jsonl",
            "basic_math_dentaku": "/storage6/fujisawa/add_ja_3x3_delete_5_pattern.jsonl",
            "aozora": "/storage6/corpus/category/BOOK/raw/JA/aozora/ja_book.jsonl",
        }
        return ratios, target_list

    ## 5/23
    if pattern_name == "pattern13":
        ratios = {
            "dolly": 1,
            "dentaku_ins": 1,
        }
        target_list = {
            "dolly": "/storage6/aa_fujimoto/tmp_dataset/databricks-dolly-15k-ja_text.jsonl",
            "dentaku_ins": "/storage6/aa_fujimoto/tmp_dataset/dentaku_instruct.jsonl",
        }
        return ratios, target_list
    # 5/24
    if pattern_name == "zoo_router":
        ratios = {
            "c4": 0.03,
            "wiki_ja": 0.1,
            "aozora": 0.4,
            "math_ins": 0.3
        }
        target_list = {
            "c4": "/storage6/dataset/pretrain/router/1B/ja_mc4/merged_mc4_6.0.jsonl",
            "wiki_ja": "/storage6/dataset/pretrain/gen_experet/WIKI/raw/wikipedia_ja/merged_wikipedia_ja_16.0.jsonl",
            "aozora": "/storage6/corpus/category/BOOK/raw/JA/aozora/ja_book.jsonl",
            "math_ins": "/storage6/corpus/category/MATH/raw/OpenMathInstruct/ja_math.jsonl",
        }
        return ratios, target_list
    if pattern_name == "zoo_sft":
        ratios = {
            "auto_qa_other": 1,
            "auto_qa_cc": 1,
            "auto_qa": 1,
            "auto_wiki_qa": 0.4,
            "bunpo_rikai": 1,
            "cotangent": 1,
            "llm_japanese": 0.1,
            "logical_datasets": 1,
        }
        target_list = {
            "auto_qa_other": "/storage6/dataset/sft/AutoGeneratedGapaneseQA-other.jsonl",
            "auto_qa_cc": "/storage6/dataset/sft/AutoGeneratedJapaneseQA-CC_unique_QA.jsonl",
            "auto_qa": "/storage6/dataset/sft/AutoGeneratedJapaneseQA_unique_QA.jsonl",
            "auto_wiki_qa": "/storage6/dataset/sft/auto_wiki_qa.jsonl",
            "bunpo_rikai": "/storage6/dataset/sft/bunpo_rikai.jsonl",
            "cotangent": "/storage6/dataset/sft/cotangent.jsonl",
            "llm_japanese": "/storage6/dataset/sft/llm_japanese.jsonl",
            "logical_datasets": "/storage6/dataset/sft/logical_datasets.jsonl",    
        }
        return ratios, target_list

    if pattern_name == "zoo_sft_small":
        ratios = {
            "auto_qa_other": 1,
            "auto_qa_cc": 1,
            "auto_qa": 1,
            "auto_wiki_qa": 0.05,
            "bunpo_rikai": 1,
            "cotangent": 1,
            "llm_japanese": 0.05,
            "logical_datasets": 1,
        }
        target_list = {
            "auto_qa_other": "/storage6/dataset/sft/AutoGeneratedGapaneseQA-other.jsonl",
            "auto_qa_cc": "/storage6/dataset/sft/AutoGeneratedJapaneseQA-CC_unique_QA.jsonl",
            "auto_qa": "/storage6/dataset/sft/AutoGeneratedJapaneseQA_unique_QA.jsonl",
            "auto_wiki_qa": "/storage6/dataset/sft/auto_wiki_qa.jsonl",
            "bunpo_rikai": "/storage6/dataset/sft/bunpo_rikai.jsonl",
            "cotangent": "/storage6/dataset/sft/cotangent.jsonl",
            "llm_japanese": "/storage6/dataset/sft/llm_japanese.jsonl",
            "logical_datasets": "/storage6/dataset/sft/logical_datasets.jsonl",
        }
        return ratios, target_list
