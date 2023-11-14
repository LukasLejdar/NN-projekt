#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H 

// activations --------------------------------------

#include <algorithm>
#include <cstddef>
#include <math.h>

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
  for(size_t i = 0; i < length; i++) {
    v[i] = exp(v[i]);
    sum += v[i];
  }
  for(size_t i = 0; i < length; i++) {
    v[i] = v[i] / sum;
  }
}

inline float crossEntropy(float *v, float *y, size_t length) {
  float sum = 0;
  for(size_t i = 0; i < length; i++) {
    sum -= y[i]*log(v[i]);
  }
  return sum;
}

#endif
