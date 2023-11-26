#include <bitset>
#include <cstdio>
#include <iostream>
#include "network/layer.hpp"
#include "network/net.hpp"
#include <iterator>
#include <thread>
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
  //training_data.number_of_entries = 10;

  const size_t CONV_LENGTH = 2;
  Convolutional conv_layers[CONV_LENGTH] {
    {{1,28,28}, {32,3,3}, {2,2}}, //input, kernel shape 
    {{32,13,13}, {64,3,3}, {2,2}},
  };

  const size_t DENSE_LENGTH = 2;
  Dense dense_layers[DENSE_LENGTH] = {
    {1600, 128},
    {128, 10},
  };

  Model model(CONV_LENGTH, conv_layers, DENSE_LENGTH, dense_layers);
  Net net(model);
  net.train_epochs(training_data, 15, test_data);
  net.test(test_data);

  return 0;
}


