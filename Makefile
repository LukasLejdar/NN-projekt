CFLAGS=-Wall -Wextra -pedantic -ggdb
CC=g++

all: _test _net _test_backprop

_test: ./test/test.cpp
	$(CC) $(CFLAGS) -o ./build/test ./test/test.cpp ./src/network/math.cpp ./src/network/net.cpp ./src/mnist_reader.cpp 

_net: ./src/main.cpp
	$(CC) $(CFLAGS) -o ./build/net ./src/main.cpp ./src/network/math.cpp ./src/network/net.cpp ./src/mnist_reader.cpp 

_test_backprop: ./test/test_backprop.cpp
	$(CC) $(CFLAGS) -o ./build/test_backprop ./test/test_backprop.cpp ./src/network/math.cpp ./src/network/net.cpp  ./src/mnist_reader.cpp 
