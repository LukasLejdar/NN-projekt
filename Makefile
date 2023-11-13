OPTIMIZATION_FLAGS=-O3 -march=native -ffast-math
CFLAGS=-Wall -Wextra -pedantic -ggdb 
CC=g++

all: _test _benchmark _net _test_backprop _paralelism

_parallelism: ./test/parallelism.cpp
	$(CC) $(CFLAGS) $(OPTIMIZATION_FLAGS) -o ./build/parallelism ./test/parallelism.cpp ./src/network/math.cpp 

_test: ./test/test.cpp
	$(CC) $(CFLAGS) -o ./build/test ./test/test.cpp ./src/network/math.cpp 

_benchmark: ./test/benchmark.cpp
	$(CC) $(CFLAGS) $(OPTIMIZATION_FLAGS) -o ./build/benchmark ./test/benchmark.cpp ./src/network/math.cpp

_net: ./src/main.cpp
	$(CC) $(CFLAGS) -o ./build/net ./src/main.cpp ./src/network/math.cpp ./src/network/net.cpp ./src/mnist_reader.cpp 

_test_backprop: ./test/test_backprop.cpp
	$(CC) $(CFLAGS) -o ./build/test_backprop ./test/test_backprop.cpp ./src/network/math.cpp ./src/network/net.cpp  ./src/mnist_reader.cpp 
