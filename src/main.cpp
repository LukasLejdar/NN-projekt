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
  MnistReader test_data("mnist/t10k-images-idx3-ubyte", "mnist/t10k-labels-idx1-ubyte");

  const size_t LENGTH = 2;
  Dense layers[LENGTH] = {
    {784, 128},
    {128, 10},
  };

  Net net(layers, LENGTH);
  net.train_epochs(training_data, 80, test_data);

  net.test(test_data);

  return 0;
}
