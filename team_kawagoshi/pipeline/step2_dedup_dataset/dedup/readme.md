# readme
deduplicationのc++実装

参考
https://github.com/HojiChar/HojiChar/blob/main/hojichar/filters/deduplication.py

https://arxiv.org/abs/2107.06499

指定したディレクトリ内のjsonlファイルに対してdedup処理を行い、指定したディレクトリに出力する

## 準備

json dump用に`nlohmann/json`を使う。  
https://github.com/nlohmann/json  

```
sudo apt install nlohmann-json3-dev
```

`simdjson`で高速にjsonlファイルを処理できるらしい

https://github.com/simdjson/simdjson

```
wget https://raw.githubusercontent.com/simdjson/simdjson/master/singleheader/simdjson.h 
wget https://raw.githubusercontent.com/simdjson/simdjson/master/singleheader/simdjson.cpp 
```


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
./deduplicate {target_dir} {output_dir}
```

# Lisence

This project is licensed under the MIT License, see the LICENSE.txt file for details