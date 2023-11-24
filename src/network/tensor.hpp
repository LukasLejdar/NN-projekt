#include <algorithm>
#include <assert.h>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <cstdio>
#include <memory>
#include <random>
#include <omp.h>
#include <math.h>
#include <strings.h>
#include <tuple>
#include <array>
#include <span>
#include <type_traits>

#ifndef MATRIX_H 
#define MATRIX_H

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Warray-bounds"

inline size_t multiply(const size_t* values, size_t count) {
  size_t result = 1;
  for (size_t i = 0; i < count; i++) result *= values[i];
  return result;
}

template<size_t dim>
struct Shape {
  size_t dimensions[dim];
  size_t size;

  Shape(): dimensions(), size(0) {} 
  Shape(const Shape& other): dimensions{}, size(other.size) {
    std::copy(other.dimensions, other.dimensions + dim, dimensions);
  }

  template<typename... Args, typename = std::enable_if_t<(sizeof...(Args) == dim)>>
  Shape(Args... args): 
    dimensions{static_cast<size_t>(args)...}, 
    size(multiply(dimensions, dim)) {} 

  void swap(Shape& other) {
    std::swap(size, other.size);
    std::swap_ranges(dimensions, dimensions+dim, other.dimensions);
  }

  size_t operator[](size_t index) {
    assert(index < dim);
    return dimensions[index];
  }

  size_t operator[](size_t index) const {
    assert(index < dim);
    return dimensions[index];
  }

  bool operator == (const Shape& other) const {
    return std::memcmp(dimensions, other.dimensions, dim* sizeof(size_t));
  }
};

template<class T, size_t dim>
struct TensorT {

  Shape<dim> shape;

  const size_t* dimensions = shape.dimensions;
  const size_t& size = shape.size;
  const size_t& ht = *(shape.dimensions+dim-2);
  const size_t& wt = *(shape.dimensions+dim-1);

  T* v;
  
  ~TensorT() { if(!is_subtensor) delete [] v; }
  TensorT(): shape(), v(nullptr) {}
  TensorT(bool is_subtensor): shape(), v(nullptr), is_subtensor(is_subtensor) {};
  TensorT(const TensorT& other): shape(other.shape), v(new T[size]) {
    std::copy(other.v, other.v+size, v);
  }

  template<typename... Args, typename = std::enable_if_t<(
      sizeof...(Args) == dim && std::is_convertible_v<T, size_t>)>>
    TensorT(Args... args) : shape(args...), v(nullptr) { 
      v = new T[size];
      std::fill(v, v+size, 0); 
    }

  TensorT(Shape<dim> shape) : shape(shape), v(new T[size]) { 
    std::fill(v, v+size, 0); 
  }

  template<typename... Args, typename = std::enable_if_t<(
      sizeof...(Args) == dim && std::is_convertible_v<T, size_t>)>>
    TensorT(float *other_v, Args... args) : shape(args...), v(nullptr) { 
      v = new T[size];
      std::copy(other_v, other_v+size, v); 
    }

  void setV(const T other_v[]) {
    std::copy(other_v, other_v+size, v);
  }

  void faltten() {
    std::fill(shape.dimensions, size, 1);
    shape.dimensions[0] = size;
  }

  void swap(TensorT& other) {
    shape.swap(other.shape);
    std::swap(other.v, v);
  }

  void zero() {
    std::fill(v, v+size, 0);
  }

  void randomize(float mean=0, float variance=0) const {
    if(variance == 0) variance = 1.0f/ht;
    std::random_device rd;
    std::mt19937 generator(rd());
    std::normal_distribution<float> distribution(mean, std::sqrt(variance));

    for(size_t i = 0; i < size; i++) v[i] = distribution(generator);
  }


  template<typename S = const TensorT>
  typename std::enable_if<(S::_dim == 1), T>::type
  const operator[](int index) const {
    return v[index];
  }

  template<typename S = const TensorT>
  typename std::enable_if<(S::_dim == 2), T*>::type
  const operator[](int index) const {
    return &v[index*wt];
  }

  template<typename S = const TensorT>
  typename std::enable_if<(S::_dim > 2), TensorT<T, dim-1>>::type
  operator[](size_t index) {
    TensorT<T, dim-1> new_tensor(true);
    std::copy(shape.dimensions+1, shape.dimensions+dim, new_tensor.shape.dimensions);
    new_tensor.v = &v[index*multiply(shape.dimensions+1, dim-1)];

    return new_tensor;
  }

  TensorT& operator*(float scaler) {
    for(size_t i = 0; i < size; i++) v[i] *= scaler;
    return *this;
  }

  TensorT& operator=(const TensorT& other) {
    //std::cout << "const assignment " << other.ht << " - " << other.wt << "\n";
    TensorT temp(other);
    swap(temp);
    return *this;
  }

  TensorT& operator=(TensorT& other) {
    //std::cout << "const assignment " << other.ht << " - " << other.wt << "\n";
    TensorT temp(other);
    swap(temp);
    return *this;
  }
  private:
    static const size_t _dim = dim;
    bool is_subtensor = false;
};

#pragma GCC diagnostic pop

template <size_t dim>
using Tensor = TensorT<float, dim>;

typedef TensorT<float, 2> Matrix;

#endif
