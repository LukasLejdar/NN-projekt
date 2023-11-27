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
  //training_data.number_of_entries = 100;

  const size_t CONV_LENGTH = 2;
  Convolutional conv_layers[CONV_LENGTH] {
    //{{1,28,28}, {1,3,3}, {2,2}}, //input, kernel shape 
    //{{1,13,13}, {1,3,3}, {2,2}},
    {{1,28,28}, {8,3,3}, {2,2}}, //input, kernel shape 
    {{8,13,13}, {16,3,3}, {2,2}}, //input, kernel shape 
  };

  const size_t DENSE_LENGTH = 2;
  Dense dense_layers[DENSE_LENGTH] = {
    {400, 128},
    {128, 10},
  };

  Model model(CONV_LENGTH, conv_layers, DENSE_LENGTH, dense_layers);
  Net net(model);
  net.train_epochs(training_data, 15, test_data);
  net.test(test_data, const_cast<char*>("accuracy for test data: "));

  return 0;
}


