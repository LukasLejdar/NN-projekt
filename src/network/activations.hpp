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

template<size_t dim>
inline void L2(const Tensor<dim>& w, const Tensor<dim>& dw, float regularization, float lerning_rate) {
  assert(dw.size == w.size);
  for(size_t i = 0; i < dw.size; i ++) {
    w.v[i] = (1-regularization)*w.v[i] - lerning_rate * dw.v[i];
  }
}

template<size_t dim>
inline void adam(const Tensor<dim>& dw, const Tensor<dim>& ema, const Tensor<dim>& ma, float decay_rate1, float decay_rate2, size_t t) {
    assert(dw.size == ema.size && dw.size == ma.size);

    for(size_t i = 0; i < dw.size; i++) {
      ma.v[i] =  decay_rate1*ma.v[i] + (1-decay_rate1)*dw.v[i];
      ema.v[i] =  decay_rate2*ema.v[i] + (1-decay_rate2)*pow(dw.v[i], 2);
      ma.v[i] = ma.v[i] / (1 - pow(decay_rate1, t));
      ema.v[i] = ema.v[i] / (1 - pow(decay_rate2, t));

      dw.v[i] = ma.v[i] / (sqrt(ema.v[i]) + 0.00000001);
    }
}

template<size_t dim>
inline void rmsProp(const Tensor<dim>& dw, const Tensor<dim>& ema, float learning_rate, float decay_rate, size_t t) {
    assert(dw.size == ema.size);

    for(size_t i = 0; i < dw.size; i++) {
      ema.v[i] =  decay_rate*ema.v[i] + (1-decay_rate)*pow(dw.v[i], 2);
      ema.v[i] = ema.v[i] / (1 - pow(decay_rate, t));

      dw.v[i] = dw.v[i]* -learning_rate/(sqrt(ema.v[i]) + 0.00000001);
    }
}

#endif
