#include <iostream>
#include <sstream>
#include <iomanip>
#include <limits>

#include "Hasher.hpp"
#include "text.hpp"
#include "./smhasher/src/MurmurHash3.h"
#include "Hasher.hpp"
#include "text.hpp"

size_t Hasher::hash(const std::string &s, uint32_t seed) const {
    uint64_t output[2];
    MurmurHash3_x64_128(s.data(), s.length(), seed, &output);
    return output[0];
}

Hasher::Hasher(size_t n, size_t n_minhash, size_t n_bucket, size_t bucket_size) 
    : N(n), N_MINHASH(n_minhash), N_BUCKET(n_bucket), BUCKET_SIZE(bucket_size) {
    if(N_MINHASH != N_BUCKET * BUCKET_SIZE) {
        std::cerr << "Error: N_MINHASH does not match N_BUCKET * BUCKET_SIZE." << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

// text myText("おはようございます");
// hasher.apply(myText2);
// for (std::string hash : myText2.getHashes()) {
//   std::cout << hash << "\n";
// }
// 83d0ba8a77610c1af76d
// 80789666bfcecdca0d33


void Hasher::apply(text &txt) const {
    std::string s = txt.getContent();
    std::vector<size_t> minHashes(N_MINHASH, std::numeric_limits<size_t>::max());

    for (size_t i = 0; i <= s.length() - N; ++i) {
        for (size_t j = 0; j < N_MINHASH; ++j) {
            minHashes[j] = std::min(minHashes[j], hash(s.substr(i, N), j));
        }
    }

    std::vector<std::string> bucketedHashes;
    for (size_t i = 0; i < N_BUCKET; ++i) {
        std::stringstream ss;
        for (size_t j = 0; j < BUCKET_SIZE; ++j) {
            ss << std::hex << std::setw(4) << std::setfill('0') << (minHashes[i * BUCKET_SIZE + j] & 0xFFFF);
        }
        bucketedHashes.push_back(ss.str());
    }
    

    txt.setHashes(bucketedHashes);
}
