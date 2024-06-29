#pragma once

#include <string>
#include <vector>

class Hasher {
private:
    size_t N;
    size_t N_MINHASH;
    size_t N_BUCKET;
    size_t BUCKET_SIZE;    
    size_t hash(const std::string &s, uint32_t seed) const;

public:
    Hasher(size_t n, size_t n_minhash, size_t n_bucket, size_t bucket_size);
    void apply(class text &txt) const;
};
