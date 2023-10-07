#include <bitset>
#include <cstdio>
#include <iostream>
#include "network/net.hpp"
#include <iterator>
#include <time.h>
#include <stdio.h>
#include <execinfo.h>
#include <signal.h>
#include <stdlib.h>
#include <unistd.h>
#include "../src/network/math.hpp"
#include "mnist_reader.hpp"

void handler(int sig) {
  void *array[10];
  size_t size;

  // get void*'s for all entries on the stack
  size = backtrace(array, 10);

  // print out all the frames to stderr
  fprintf(stderr, "Error: signal %d:\n", sig);
  backtrace_symbols_fd(array, size, STDERR_FILENO);
  exit(1);
}

int main() {
  srand ( time(NULL) );
  signal(SIGSEGV, handler);
  signal(SIGABRT, handler);
  
  MnistReader training_data("mnist/train-images-idx3-ubyte", "mnist/train-labels-idx1-ubyte");
  
  Dense layers[] = {
    DENSE(784, 128, RELU),
    DENSE(128, 10, SOFTMAX),
  };

  Net net(layers, 2);
  net.train_epochs(training_data, 1);

  MnistReader test_data("mnist/t10k-images-idx3-ubyte", "mnist/t10k-labels-idx1-ubyte");
  net.test(test_data, 0);

  return 0;
}
