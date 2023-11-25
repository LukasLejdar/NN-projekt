OPTIMIZATION_FLAGS=-O3 -march=native -ffast-math 
CFLAGS=-Wall -Wextra -pedantic -ggdb 
CC=g++

all: _test _benchmark _net _test_backprop _paralelism _net_debug
	mkdir -p build

_parallelism: ./test/parallelism.cpp
	mkdir -p build
	$(CC) $(CFLAGS) $(OPTIMIZATION_FLAGS) -o ./build/parallelism ./test/parallelism.cpp ./src/network/math.cpp

_test: ./test/test.cpp
	mkdir -p build
	$(CC) $(CFLAGS)  -o ./build/test ./test/test.cpp ./src/network/math.cpp ./src/mnist_reader.cpp

_benchmark: ./test/benchmark.cpp
	mkdir -p build
	$(CC) $(CFLAGS) $(OPTIMIZATION_FLAGS) -DNDEBUG -o ./build/benchmark ./test/benchmark.cpp ./src/network/math.cpp

_net: ./src/main.cpp
	mkdir -p build
	$(CC) $(CFLAGS) $(OPTIMIZATION_FLAGS) -o ./build/net ./src/main.cpp ./src/network/math.cpp ./src/network/net.cpp ./src/mnist_reader.cpp ./src/network/layer.cpp

_net_debug: ./src/main.cpp
	mkdir -p build
	$(CC) $(CFLAGS) -o ./build/net_debug ./src/main.cpp ./src/network/math.cpp ./src/network/net.cpp ./src/mnist_reader.cpp ./src/network/layer.cpp

_test_backprop: ./test/test_backprop.cpp
	mkdir -p build
	$(CC) $(CFLAGS) -o ./build/test_backprop ./test/test_backprop.cpp ./src/network/math.cpp ./src/network/net.cpp  ./src/mnist_reader.cpp ./src/network/layer.cpp
