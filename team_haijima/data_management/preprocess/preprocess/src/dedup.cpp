#include <string>
#include <vector>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <limits>
#include "../smhasher/src/MurmurHash3.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>


class text {
private:
    std::string content;    
    std::vector<std::string> hashes;
public:
    text(const std::string &s);

    void setHashes(const std::vector<std::string>& h);
    std::vector<std::string> getHashes() const;
    std::string getContent() const;
};


text::text(const std::string &s) : content(s) {}

void text::setHashes(const std::vector<std::string>& h) {
    hashes = h;
}

std::vector<std::string> text::getHashes() const {
    return hashes;
}

std::string text::getContent() const {
    return content;
}



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


size_t Hasher::hash(const std::string &s, uint32_t seed) const {
    uint32_t output;
    MurmurHash3_x86_32(s.data(), s.length(), seed, &output);
    //std::cout << s.data() << ":" << s.length() << std::endl;
    //MurmurHash3_x86_32(s.data(), 5, seed, &output);
    return output;
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
        ss << i << "+";
        for (size_t j = 0; j < BUCKET_SIZE; ++j) {
            ss << std::hex << std::setw(4) << std::setfill('0') << (minHashes[i * BUCKET_SIZE + j] & 0xFFFF);
        }
        bucketedHashes.push_back(ss.str());
    }
    

    txt.setHashes(bucketedHashes);
}

namespace py = pybind11;

PYBIND11_MODULE(dedup, m) {
    py::class_<Hasher>(m, "Hasher")
        .def(py::init<size_t, size_t, size_t, size_t>())
        .def("apply", &Hasher::apply);
    py::class_<text>(m, "text")
        .def(py::init<const std::string &>())
        .def("setHashes", &text::setHashes)
        .def("getHashes", &text::getHashes)
        .def("getContent", &text::getContent);
}

