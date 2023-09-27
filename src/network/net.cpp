#include <cstring>
#include <string>
#include "net.hpp"

void initialize_layer(Dense dense) {
  float weights[dense.in_shape*dense.out_shape];
  memset(weights, 0, dense.in_shape*dense.out_shape*sizeof(float));
  dense.w = weights;

  float biases[dense.out_shape];
  memset(biases, 0, dense.out_shape*sizeof(float));
  dense.b = biases;
}

Net::Net(int in_shapes[], int length) {
  Dense layers[length];
  memset(layers, 0, length*sizeof(Dense));

  for(int i = 0; i < length-1; i++) {
    layers[i].in_shape = in_shapes[i];
    layers[i].out_shape = in_shapes[i+1];
  }

  for(int i = 1; i < length; i++) {
    initialize_layer(layers[i]);
  }

}

void Net::forward_prop(float v[]) {
  return;
}



