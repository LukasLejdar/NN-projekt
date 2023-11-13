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

int main() {
  MnistReader training_data("mnist/train-images-idx3-ubyte", "mnist/train-labels-idx1-ubyte");

  const size_t LENGTH = 3;
  Dense layers[LENGTH] = {
    {784, 120},
    {120, 84},
    {84, 10},
  };

  Net net(layers, LENGTH);
  Matrix X(784, 1);
  Matrix Y(10, 1);

  net.train_epochs(training_data, 200);

  //MnistReader test_data("mnist/t10k-images-idx3-ubyte", "mnist/t10k-labels-idx1-ubyte");
  //net.test(test_data);

  return 0;
}
