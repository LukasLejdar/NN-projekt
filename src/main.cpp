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
  signal(SIGSEGV, handler);   // install our handler
  signal(SIGABRT, handler);   // install our handler
  
  MnistReader training_data("mnist/train-images-idx3-ubyte", "mnist/train-labels-idx1-ubyte");

  for(int i = 0; i < 11; i++) {
    training_data.read_next();

    std::cout << "\n" << training_data.last_lable << "\n";
    drawMat(training_data.last_read);
    ;
  }

  return 0;
}
