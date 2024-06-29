## 重複削除

# readme
deduplicationのc++実装

参考
https://github.com/HojiChar/HojiChar/blob/main/hojichar/filters/deduplication.py
https://github.com/if001/dedup_sentence.git

https://arxiv.org/abs/2107.06499

指定したディレクトリ内のjsonlファイルに対してdedup処理を行い、指定したディレクトリに出力する

## 準備

文章のハッシュ計算用に`MurmurHash`を使う。  
https://github.com/aappleby/smhasher/blob/master/src/MurmurHash3.cpp

```
git clone https://github.com/aappleby/smhasher.git
```

```
make
```

## run

```
./deduplicate {target_dir} {output_dir} {num_process}
```

テキストを分割する場合は、sep.sh内のinput_jsonとoutput_dirを任意に設定し、下記を実行。
```
bash ./sep.sh
```

slurmにて実行する場合は、下記を実行。
```
sbatch dedup_slurm.sh
```

## tips

main.cpp 31行目の下記コードを、変更することで重複削除の精度と速度を変更可能。
```
    Hasher hasher(10, 4, 2, 2);
```
左から順に、N-GRAM, N_BUCKET*BUCKET_SIZE, N_BUCKET, BUCKET_SIZEとなっている。

# Lisence

This project is licensed under the MIT License, see the LICENSE.txt file for details