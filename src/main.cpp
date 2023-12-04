#include <algorithm>
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
  MnistReader monstrous_set(training_set, 0, 120);
  std::copy(monsters_ids, monsters_ids + NMONSTERS, monstrous_set.permutation.v);

  const size_t CONV_LENGTH = 1;
  Convolutional conv_layers[CONV_LENGTH] {
    {{1,28,28}, {4,3,3}, {2,2}}, //input, kernel shape 
  };

  const size_t DENSE_LENGTH = 2;
  Dense dense_layers[DENSE_LENGTH] = {
    {676, 128},
    {128, 2},
  };

  Model model(CONV_LENGTH, conv_layers, DENSE_LENGTH, dense_layers);
  Net net(model);
  net.train_epochs(monstrous_set, 10, 1);

  std::iota(training_set.permutation.v, training_set.permutation.v + training_set.permutation.size, 0);

  net.make_preds(training_set.getAllImages(), "train_predictions.csv");

  return 0;
}


