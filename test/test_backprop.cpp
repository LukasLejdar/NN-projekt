#include <algorithm>
#include <assert.h>
#include <cstdio>
#include <iostream>
#include "test.hpp"
#include "../src/network/math.hpp"
#include "../src/network/net.hpp"
#include "../src/network/activations.hpp"

//Params are expected to be one of the weights, biases, or kernels in cache,
//though if they are not, it will calculate the gradient anyway. It ill be zero.
template<size_t dim>
Tensor<dim> get_true_grad(Cache& cache, Tensor<dim>& params) {
  const Tensor<dim> true_dParams(params.shape);
  const Tensor<dim> params_orig(params.shape);
  copyToTensorOfSameSize(params, params_orig);

  for(size_t i = 0; i < params.size; i++) {
    float e = 0.001;

    params.v[i] += e;
    forward_prop(cache);
    float entropy1 = crossEntropy(cache.dense.a[cache.dense.count-1].v, cache.y);

    params.v[i] -= 2*e;
    forward_prop(cache);
    float entropy2 = crossEntropy(cache.dense.a[cache.dense.count-1].v, cache.y);

    true_dParams.v[i] = (entropy1 - entropy2) /(2*e);
    params.v[i] += e;
  }
  
  copyToTensorOfSameSize(params_orig, params);
  return true_dParams;
}

int main() {
  MnistReader training_set("data/fashion_mnist_train_vectors.csv", "data/fashion_mnist_train_labels.csv", {28,28}, 60000);
  training_set.number_of_entries = 300;
  training_set.shuffle();

  const size_t CONV_LENGTH = 2;
  Convolutional conv_layers[CONV_LENGTH] {
    {{1,28,28}, {4,3,3}, {2,2}}, //input, kernel shape 
    {{4,13,13}, {8,3,3}, {3,3}}, //input, kernel shape 
  };

  const size_t DENSE_LENGTH = 1;
  Dense dense_layers[DENSE_LENGTH] = {
    {128, 10},
  };

  Model model(CONV_LENGTH, conv_layers, DENSE_LENGTH, dense_layers);
  Net net(model);

  Cache cache;
  initialize_cache(cache, model);
  net.train(cache, training_set, 0, 0);

  training_set.loop_to_beg();
  training_set.read_next(false);
  zeroGradients(cache);
  net.prepare_cache(training_set.last_read, training_set.last_lable, cache);

  float entropy = 0;
  while (entropy < 0.1) {
    forward_prop(cache);
    back_prop(cache);
    entropy = crossEntropy(cache.dense.a[cache.dense.count-1].v, cache.y);
  }

  drawConv(cache);

  for(size_t i = 0; i < CONV_LENGTH; i++) {
    Tensor<4> true_dK = get_true_grad(cache, cache.conv.k[i]);
    std::cout << "\ncalculated dK " << i << "\n";
    drawKernels(cache.conv.dK[i]);
    std::cout << "\ntrue dK " << i << "\n";
    drawKernels(true_dK);

    Tensor<3> true_dB = get_true_grad(cache, cache.conv.b[i]);
    std::cout << "\ncalculated dB " << i << "\n";
    draw3D(cache.conv.dB[i]);
    std::cout << "\ntrue dB " << i << "\n";
    draw3D(true_dB);
  }

  return 0;
}


