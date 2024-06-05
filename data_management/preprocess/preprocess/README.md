# Data Preprocessor

## Install
```bash
pip install cython pybind11
git clone https://github.com/aappleby/smhasher.git
pip install .
```

## How to use
```python
from preprocess import dedup
from preprocess import filters

filters.is_japanese('吾輩は猫である。')  # => True

hasher = dedup.Hasher(10, 200, 20, 10)  # N_gram, n_minhash, n_buckets, bucket_size
text = dedup.text('吾輩は猫である。')
hasher.apply(text)
text.getHashes()
# => 
# ['0+5925bfeee1aac5371dfdd005cc1fd0b4df1cd5cc',
# '1+6b08f077390d2501f0fe1f8272546a6b85fa0c29',
# '2+99663c4aa080b519002363ed38a09405d5016f0a',
# '3+a2f8142479b7aae221a29f85c811435e44cf3000',
# '4+0904f69bb7f043f44a2a8fe89df9e1d432fbc5db',
# '5+844952e9bc7462c1a7286847aaf8a7f8a400b15b',
# '6+ae3b1c26c8b7699f3eecc4b823f1bbf8bcd54118',
# '7+2891386a4e74d9c526835a456d6906d4a150130b',
# '8+b62fc164b309b162b751ac41af7fcee946371e6b',
# '9+b74c175dc70819fb833010e6243c0fd51f235678',
# '10+2df5891620f403820214e1230ecf0f1962516787',
# '11+d718998299e9215c9eb89a95fe12febfa49f7f0e',
# '12+11c7591042801fe6a241270ecc716de7ca5e1923',
# '13+bcc04c00ab7829a97e364a38c5c568c8fcca0d58',
# '14+7aebef76f5652207a472fa54e543756b9f488af6',
# '15+45db0149b57323faaed4219c2aa4eda03099dd12',
# '16+b521084671befc67f04d0eabf002a7dd1dd606db',
# '17+491a5cda7c5d816e06b6321b0b2116530c558fa5',
# '18+fdc88a09a576131a8fa2089666cf7c65a1f554da',
# '19+e1715f71d16b65b8b1e5499c84d2ad7960002ac6']
```

