#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H 

// activations --------------------------------------

#include <algorithm>
#include <cstddef>
#include <math.h>
#include <stdexcept>

inline void relu(float v[], size_t length) {
  for(size_t i = 0; i < length; i++) {
    v[i]=std::max<float>(0.0, v[i]);
  }
}

inline void relu_backward(float dA[], float A[], size_t length) {
  for(size_t i = 0; i < length; i++) {
    dA[i] = (A[i] > 0) * dA[i];
  }
}

inline void sigmoid(float v[], size_t length) {
  for(size_t i = 0; i < length; i++) {
    v[i] = 1 / (1 + exp(-v[i]));
  }
}

inline void softmax(float v[], size_t length) {
  float sum = 0;
  float max = *std::max_element(v, v + length);
  for(size_t i = 0; i < length; i++) {
    v[i] -= max;
    v[i] = exp(v[i]);
    sum += v[i];
  }
  for(size_t i = 0; i < length; i++) {
    v[i] = v[i] / sum;
  }
}

inline float crossEntropy(float *v, size_t y) {
  return -log(v[y] + 0.00000001);
}

#endif
