#include <algorithm>
#include <assert.h>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <cstdio>
#include <memory>
#include <random>
#include <omp.h>
#include <math.h>
#include <tuple>
#include <array>
#include <span>
#include <type_traits>

#ifndef MATRIX_H 
#define MATRIX_H

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Warray-bounds"

inline size_t multiply(size_t* values, size_t count) {
  size_t result = 1;
  for (size_t i = 0; i < count; i++) result *= values[i];
  return result;
}

template<class T, size_t dim>
struct Tensor {
  size_t dimensions[dim];
  size_t size = 0;
  T* v;
  
  size_t depth = dimensions[dim-3];
  size_t ht = dimensions[dim-2];
  size_t wt = dimensions[dim-1];

  ~Tensor() { if(!is_subtensor) delete [] v; }

  Tensor(): dimensions{}, size(0), v(nullptr) {}

  Tensor(bool is_subtensor): dimensions{}, v(nullptr), is_subtensor(is_subtensor) {};

  Tensor(const Tensor& other): dimensions(), size(other.size), v(new T[size]) {
    std::copy(other.dimensions, other.dimensions + dim, dimensions);
    std::copy(other.v, other.v+size, v);
  }

  template<typename... Args, typename = std::enable_if_t<(sizeof...(Args) == dim && dim < 1000 )>>
    Tensor(Args... args) : 
      dimensions{static_cast<size_t>(args)...}, 
      size(multiply(dimensions, dim)), 
      v(new T[size]) { std::fill(v, v+size, 0); }

  template<typename... Args, typename = std::enable_if_t<sizeof...(Args) == dim>>
    Tensor(float *other_v, Args... args) : 
      dimensions{static_cast<size_t>(args)...}, 
      size(multiply(dimensions, dim)), 
      v(new T[size]) { std::copy(other_v, other_v+size, v); }

  void setV(const T other_v[]) {
    std::copy(other_v, other_v+size, v);
  }

  void faltten() {
    std::fill(dimensions, size, 1);
    dimensions[0] = size;
  }

  void swap(Tensor& other) {
    for(size_t i = 0; i < dim; i++) 
      std::swap(other.dimensions[i], dimensions[i]);
    std::swap(other.v, v);
  }

  template<typename S = Tensor>
  typename std::enable_if<(S::_dim == 1), T>::type
  operator[](int index) {
    return v[index];
  }

  template<typename S = Tensor>
  typename std::enable_if<(S::_dim == 2), T*>::type
  operator[](int index) {
    return &v[index*wt];
  }

  template<typename S = Tensor>
  typename std::enable_if<(S::_dim > 2), Tensor<T, dim-1>>::type
  operator[](size_t index) {
    Tensor<T, dim-1> tensor(true);
    std::copy(dimensions+1, dimensions+dim-1, tensor.dimensions);
    tensor.v = &v[index*dimensions[0]];

    return tensor;
  }

  Tensor& operator*(float scaler) {
    for(size_t i = 0; i < size; i++) v[i] *= scaler;
    return *this;
  }

  Tensor& operator=(const Tensor& other) {
    //std::cout << "const assignment " << other.ht << " - " << other.wt << "\n";
    Tensor temp(other);
    swap(temp);
    return *this;
  }

  Tensor& operator=(Tensor& other) {
    //std::cout << "const assignment " << other.ht << " - " << other.wt << "\n";
    Tensor temp(other);
    swap(temp);
    return *this;
  }
  private:
    static const size_t _dim = dim;
    bool is_subtensor = false;
};

#pragma GCC diagnostic pop

typedef Tensor<float, 2> Matrix;

#endif
