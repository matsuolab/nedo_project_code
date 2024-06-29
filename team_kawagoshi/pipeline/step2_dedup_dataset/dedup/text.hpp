#pragma once

#include <string>
#include <vector>

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
