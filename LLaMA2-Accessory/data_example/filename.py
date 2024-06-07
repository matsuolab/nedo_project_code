import json

# ファイル名を格納するリストを作成
#file_names = [f"gsm8k_splitted_{i:05d}.parquet" for i in range(0,102)]
file_names = [f"add0520_shuffled_split_{i:04d}.parquet" for i in range(0,460)]
#file_names = [f"en1029_ja8301_shuffled_split_{i:04d}.parquet" for i in range(0,9323)]

# JSONファイルに書き込む
with open('file_list_honban_add.json', 'w') as f:
    json.dump(file_names, f,indent=2)

print("JSONファイルが作成されました。")
