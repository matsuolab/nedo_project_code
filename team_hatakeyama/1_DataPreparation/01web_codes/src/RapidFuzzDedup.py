from datetime import datetime, timedelta
from rapidfuzz.process import cdist
import random
import glob
import os
import json


def get_deduped_ids(similarity_scores, threshold=90):
    deduped_ids = []
    ignore_indices = set()  # 類似度が高く無視すべき行のインデックス
    # 類似度閾値

    # 各行についてチェック
    for i in range(len(similarity_scores)):
        if i not in ignore_indices:  # この行がまだ処理されていない場合
            deduped_ids.append(i)
            # 類似度が閾値以上のすべての行を無視リストに追加
            for j in range(len(similarity_scores)):
                if similarity_scores[i][j] >= threshold and i != j:
                    ignore_indices.add(j)
    return deduped_ids


def dedup_lines(check_lines, check_length=100, n_workers=16, threshold=90):
    original_lines = check_lines

    check_lines = []
    start_pos = random.randint(0, 300)
    for line in original_lines:
        if len(line) < check_length+start_pos:
            check_lines.append(line[:check_length])
        else:
            check_lines.append(line[start_pos:start_pos+check_length])
    similarity_scores = cdist(
        check_lines, check_lines, workers=n_workers)
    deduped_ids = get_deduped_ids(similarity_scores, threshold=threshold)
    deduped_lines = [original_lines[i] for i in deduped_ids]
    return deduped_lines


def dedup_dir(cluster_id,
              check_length=100,  # 類似度判定の長さ
              check_n=2000,  # 類似度判定のバッチサイズ
              n_workers=32,  # 類似度判定の並列数
              threshold=35,  # 類似度の閾値｡低めにすると､似た文章をより厳しく弾ける
              save_batch_size=1000,  # 重複削除後のファイルの保存バッチサイズ
              repetition=2,  # バッチ化した際の繰り返し回数
              ):

    print("Cluster ID: ", cluster_id)
    path_list = glob.glob(f"../data/categorized/{cluster_id}/*.jsonl")


    #if os.path.exists(f"../data/dedup_categorized/{cluster_id}"):
    if len(glob.glob(f"../data/dedup_categorized/{cluster_id}/*.jsonl")) > 0:
        print("Already deduped")
        return

    all_lines = []
    for path in path_list:
        try:
            with open(path, "r") as f:
                lines = f.readlines()
            lines = [i[10:-3] for i in lines]
            all_lines += lines
        except Exception as e:
            print(e)
            print("Error in ", path)

    # 普通の重複検出
    all_lines = list(set(all_lines))
    print("start: ",len(all_lines))
    n_repeat = len(all_lines)//check_n*repetition
    n_repeat = 1 if n_repeat == 0 else n_repeat

    # 類似度判定｡ 速度がバッチサイズの二乗で落ちるので､バッチサイズを小さくして､ランダムに落していく
    for i in range(n_repeat):
        random.shuffle(all_lines)
        check_lines = all_lines[:check_n]
        deduped_lines = dedup_lines(
            check_lines, check_length=check_length, n_workers=n_workers, threshold=threshold)
        all_lines = all_lines[check_n:]+deduped_lines
        print(len(all_lines))

    if not os.path.exists(f"../data/dedup_categorized"):
        os.makedirs(f"../data/dedup_categorized")
    if not os.path.exists(f"../data/dedup_categorized/{cluster_id}"):
        os.makedirs(f"../data/dedup_categorized/{cluster_id}")

    cnt = 0
    # save_batch_size件ずつ､ファイルに保存
    for i in range(0, len(all_lines), save_batch_size):
        with open(f"../data/dedup_categorized/{cluster_id}/{cnt}.jsonl", "w") as f:
            for line in all_lines[i:i+save_batch_size]:
                f.write(json.dumps({"text": line}, ensure_ascii=False)+"\n")
        cnt += 1
