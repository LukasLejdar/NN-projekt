#include <cstdio>
#include <iostream>
#include "network/net.hpp"
#include <time.h>
#include <stdio.h>
#include <execinfo.h>
#include <signal.h>
#include <stdlib.h>
#include <unistd.h>
#include "../src/network/math.hpp"

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


  Dense* layers{new Dense[]{
    DENSE(5, 4, RELU),
    DENSE(4, 2, SOFTMAX),
  }};

  Net net(layers, 2);
  float inv[] = {1,1,1,1,1};
  Matrix input = {5, 1, inv};
  for(int i = 0; i < 10; i++) {
    net.train(input, 1, 0);
  }

  return 0;
}
