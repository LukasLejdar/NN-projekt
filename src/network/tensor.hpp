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

inline size_t multiply(size_t* values, size_t count) {
  size_t result = 1;
  for (size_t i = 0; i < count; i++) result *= values[i];
  return result;
}

template<size_t dim>
struct Shape {
  size_t dimensions[dim];
  size_t size;

  size_t depth = dimensions[dim-3];
  size_t ht = dimensions[dim-2];
  size_t wt = dimensions[dim-1];

  Shape(): dimensions{}, size(0) {} 
  Shape(const Shape& other): size(other.size) {
    std::copy(other.dimensions, other.dimensions + dim, dimensions);
  }

  template<typename... Args, typename = std::enable_if_t<(sizeof...(Args) == dim)>>
  Shape(Args... args): 
    dimensions{static_cast<size_t>(args)...}, 
    size(multiply(dimensions, dim)) {} 

  void swap(Shape& other) {
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

  const size_t (&dimensions)[dim] = dimensions;
  const size_t &size = shape.size;
  const size_t &wt = shape.wt;
  const size_t &ht = shape.ht;
  const size_t &depth = shape.depth;

  T* v;
  
  ~TensorT() { if(!is_subtensor) delete [] v; }
  TensorT(): shape(), v(nullptr) {}
  TensorT(bool is_subtensor): shape(), v(nullptr), is_subtensor(is_subtensor) {};
  TensorT(const TensorT& other): shape(other.shape), v(new T[size]) {
    std::copy(other.v, other.v+size, v);
  }

  template<typename... Args, typename = std::enable_if_t<(sizeof...(Args) == dim)>>
    TensorT(Args... args) : shape(args...), v(new T[size]) { 
      std::fill(v, v+size, 0); 
    }

  TensorT(Shape<dim> shape) : shape(shape), v(new T[size]) { 
    std::fill(v, v+size, 0); 
  }

  template<typename... Args, typename = std::enable_if_t<sizeof...(Args) == dim>>
    TensorT(float *other_v, Args... args) : shape(args...), v(new T[size]) { 
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
  const operator[](size_t index) const {
    TensorT<T, dim-1> tensor(true);
    std::copy(dimensions+1, dimensions+dim-1, tensor.dimensions);
    tensor.v = &v[index*dimensions[0]];

    return tensor;
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
