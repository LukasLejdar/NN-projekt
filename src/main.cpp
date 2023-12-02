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
  MnistReader training_set("data/fashion_mnist_train_vectors.csv", "data/fashion_mnist_train_labels.csv", {28,28}, 60000);

  const size_t CONV_LENGTH = 1;
  Convolutional conv_layers[CONV_LENGTH] {
    {{1,28,28}, {32,3,3}, {2,2}}, //input, kernel shape 
  };

  const size_t DENSE_LENGTH = 2;
  Dense dense_layers[DENSE_LENGTH] = {
    {5408, 128},
    {128, 10},
  };

  Model model(CONV_LENGTH, conv_layers, DENSE_LENGTH, dense_layers);
  Net net(model);
  net.train_epochs(training_set, 40, 0.921);

  MnistReader test_set("data/fashion_mnist_test_vectors.csv", "data/fashion_mnist_test_labels.csv", {28,28}, 10000);
  net.test(test_set, const_cast<char*>("accuracy for test data: "));
  net.make_preds(test_set.getAllImages(), "test_predictions.csv");
  net.make_preds(training_set.getAllImages(), "train_predictions.csv");

  return 0;
}


