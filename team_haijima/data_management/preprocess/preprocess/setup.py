from setuptools import setup, Extension
from Cython.Build import cythonize
import pybind11

ext1 = Extension(
    'preprocess.dedup',
    ['src/dedup.cpp', 'smhasher/src/MurmurHash3.cpp'],
    include_dirs=[pybind11.get_include(), 'smhasher/src'],
    extra_compile_args=['-std=c++17', '-Wall', '-Wextra'],
    libraries=['stdc++fs'],
    language='c++'
)

ext2 = Extension(
    'preprocess.filters',
    ['src/filters.pyx']
)
ext2 = cythonize(ext2)[0]

setup(
    name='preprocess',
    version='0.1',
    ext_modules=[ext1, ext2]
)
