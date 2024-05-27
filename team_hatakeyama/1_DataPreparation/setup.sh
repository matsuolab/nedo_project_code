conda create -n textprocess -y
conda activate textprocess
conda install pip -y
pip install datasets==2.18.0

#mecab 日本語の解析に使います
sudo apt install mecab -y
sudo apt install libmecab-dev -y
sudo apt install mecab-ipadic-utf8 -y
pip install mecab==0.996.3 
pip install ja-sentence-segmenter==0.0.2
pip install hojichar==0.9.0

#text clustering
pip install gensim==4.3.2
pip install scikit-learn==1.4.1.post1 
#pip install dask==2024.3.1
#pip install "dask[dataframe]"
#pip install jinja2-3.1.3

#model download (w2v)
cd data
mkdir model
cd model
wget http://www.cl.ecei.tohoku.ac.jp/~m-suzuki/jawiki_vector/data/20170201.tar.bz2
tar -xjvf 20170201.tar.bz2

cd ../../


#dedup programの build
sudo apt install nlohmann-json3-dev -y

git clone https://github.com/if001/dedup_sentence
cd dedup_sentence
git clone https://github.com/aappleby/smhasher.git
wget https://raw.githubusercontent.com/simdjson/simdjson/master/singleheader/simdjson.h 
wget https://raw.githubusercontent.com/simdjson/simdjson/master/singleheader/simdjson.cpp 
make