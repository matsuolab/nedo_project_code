CXX=g++
CXXFLAGS=-std=c++17 -Wall -Wextra -Wno-implicit-fallthrough -I. -pthread
LDFLAGS=-lstdc++fs -pthread

all: deduplicate clean

deduplicate: main.o Hasher.o text.o MurmurHash3.o simdjson.o
	$(CXX) $(CXXFLAGS) -o deduplicate main.o Hasher.o text.o MurmurHash3.o simdjson.o $(LDFLAGS) $(LDLIBS)

main.o: main.cpp
	$(CXX) $(CXXFLAGS) -c main.cpp

Hasher.o: Hasher.cpp Hasher.hpp
	$(CXX) $(CXXFLAGS) -c Hasher.cpp

text.o: text.cpp text.hpp
	$(CXX) $(CXXFLAGS) -c text.cpp

MurmurHash3.o: smhasher/src/MurmurHash3.cpp
	$(CXX) $(CXXFLAGS) -c smhasher/src/MurmurHash3.cpp

simdjson.o: simdjson.cpp simdjson.h
	$(CXX) $(CXXFLAGS) -c ./simdjson.cpp

clean:
	rm -f *.o 
