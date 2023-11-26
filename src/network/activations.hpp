#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H 

// activations --------------------------------------

#include <algorithm>
#include <cstddef>
#include <math.h>
#include <stdexcept>
#include "math.hpp"

template<size_t dim>
inline void relu(Tensor<dim>& tensor) {
  for(size_t i = 0; i < tensor.size; i++) {
    tensor.v[i]=std::max<float>(0.0, tensor.v[i]);
  }
}

template<size_t dim>
inline void relu_backward(Tensor<dim>& dA, Tensor<dim>& A) {
  assert(dA.size == A.size);
  for(size_t i = 0; i < A.size; i++) {
    dA.v[i] = (A.v[i] > 0) * dA.v[i];
  }
}

template<size_t dim>
inline void softmax(Tensor<dim>& t) {
  float sum = 0;
  float max = *std::max_element(t.v, t.v + t.size);
  for(size_t i = 0; i < t.size; i++) {
    t.v[i] -= max;
    t.v[i] = exp(t.v[i]);
    sum += t.v[i];
  }
  for(size_t i = 0; i < t.size; i++) {
    t.v[i] = t.v[i] / sum;
  }
}

inline float crossEntropy(float *v, size_t y) {
  return -log(v[y] + 0.00000001);
}

#endif
