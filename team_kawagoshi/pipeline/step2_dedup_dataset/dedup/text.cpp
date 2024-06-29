#include "text.hpp"

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
