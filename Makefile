CFLAGS=-Wall -Wextra -pedantic -ggdb
CC=g++

all: _test _net

_test: ./test/test.cpp
	$(CC) $(CFLAGS) -o ./bin/test ./test/test.cpp ./src/network/math.cpp ./src/network/net.cpp

_net: ./src/main.cpp
	$(CC) $(CFLAGS) -o ./bin/net ./src/main.cpp ./src/network/math.cpp ./src/network/net.cpp

