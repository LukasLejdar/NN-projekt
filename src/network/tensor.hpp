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

template <size_t dim, typename = void>
struct Shape;

template<class T, size_t dim, typename = void>
struct TensorT; 

template<size_t dim>
struct Shape<dim, std::enable_if_t<(dim > 1)>> {
  size_t dimensions[dim];
  size_t size;

  const size_t& ht = *(dimensions+dim-2);
  const size_t& wt = *(dimensions+dim-1);

  Shape(): dimensions{}, size(0) {} 
  Shape(const Shape& other): dimensions(), size(other.size) {
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

  void reshape(const size_t dims[dim]) {
    std::copy(dims, dims + dim, dimensions);
    size = multiply(dims, dim);
  }

  size_t operator[](size_t index) {
    assert(index < dim);
    return dimensions[index];
  }

  size_t operator[](size_t index) const {
    assert(index < dim);
    return dimensions[index];
  }

  Shape& operator=(const Shape& other) {
    Shape temp(other);
    swap(temp);
    return *this;
  }

  bool operator==(const Shape& other) const {
    return std::equal(dimensions, dimensions+dim, other.dimensions, other.dimensions+dim);
  }
};

template<class T, size_t dim>
struct TensorT<T, dim, std::enable_if_t<(dim > 1)>> {
  Shape<dim> shape;

  const size_t* dimensions = shape.dimensions;
  const size_t& size = shape.size;
  const size_t& ht = *(shape.dimensions+dim-2);
  const size_t& wt = *(shape.dimensions+dim-1);

  T* v;
  
  ~TensorT() { 
    if(!is_subtensor) {
      //std::cout << "deleting " << ht << " - " << wt << "\n";
      delete [] v; 
    }
  }

  TensorT(): shape(), v(nullptr), is_subtensor(false) {}
  TensorT(bool is_subtensor): shape(), v(nullptr), is_subtensor(is_subtensor){};
  TensorT(const TensorT& other): shape(other.shape), v(new T[size]), is_subtensor(false) {
    std::copy(other.v, other.v+size, v);
  }

  template<typename... Args, typename = std::enable_if_t<(
      sizeof...(Args) == dim && std::is_convertible_v<T, size_t>)>>
    TensorT(Args... args) : shape(args...), v(nullptr), is_subtensor(false) { 
      v = new T[size];
      std::fill(v, v+size, 0); 
    }

  TensorT(Shape<dim> shape) : shape(shape), v(new T[size]), is_subtensor(false) { 
    std::fill(v, v+size, 0); 
  }

  template<typename... Args, typename = std::enable_if_t<(
      sizeof...(Args) == dim && std::is_convertible_v<T, size_t>)>>
    TensorT(float *other_v, Args... args) : shape(args...), v(nullptr), is_subtensor(false) { 
      v = new T[size];
      std::copy(other_v, other_v+size, v); 
    }

  void faltten() {
    std::fill(shape.dimensions, size, 1);
    shape.dimensions[0] = size;
  }

  TensorT<T, dim-1> operator[](size_t index) const {
    //std::cout << "[] operator on " << ht << " - " << wt << "\n";
    TensorT<T, dim-1> new_tensor(true);
    new_tensor.shape.reshape(dimensions+1);
    new_tensor.v = &v[index*new_tensor.size];

    return new_tensor;
  }

  TensorT<T, 1> vectorize() {
    //std::cout << "vectorizing " << ht << " - " << wt << "\n";
    TensorT<T, 1> new_tensor(true);
    new_tensor.shape.reshape(&size);
    new_tensor.v = v;

    return new_tensor;
  }

  void vectorize(TensorT<T, 1>& V) {
    V.shape.reshape(&size);
    V.set_is_subtensor(true);
    delete [] V.v;
    V.v = v;
  }

  TensorT& operator*(T scaler) {
    for(size_t i = 0; i < size; i++) v[i] *= scaler;
    return *this;
  }

  void swap(TensorT& other) {
    shape.swap(other.shape);
    std::swap(other.v, v);
    std::swap(other.is_subtensor, is_subtensor);
  }

  TensorT& operator=(const TensorT& other) {
    TensorT temp = TensorT(other);
    swap(temp);
    return *this;
  }

  TensorT& operator=(TensorT& other) {
    swap(other);
    return *this;
  }

  void set_is_subtensor(bool value) {
    is_subtensor = value;
  }

  private:
    static const size_t _dim = dim;
    bool is_subtensor;

};

template<size_t dim>
struct Shape<dim, std::enable_if_t<(dim == 1)>> {
  size_t size;

  Shape(): size(0) {} 
  Shape(const Shape& other): size(other.size) {}
  Shape(size_t size): size(size) {}

  void swap(Shape& other) {
    std::swap(size, other.size);
  }

  void reshape(const size_t dims[dim]) {
    size = dims[0];
  }

  size_t operator[](size_t index) {
    assert(index == 0);
    return size;
  }

  size_t operator[](size_t index) const {
    assert(index == dim);
    return size;
  }

  bool operator==(const Shape& other) const {
    return size == other.size;
  }
};

template<class T, size_t dim>
struct TensorT<T, dim, std::enable_if_t<dim == 1>> {
  Shape<dim> shape;
  const size_t& size = shape.size;
  T* v;
  
  ~TensorT() { 
    if(!is_subtensor) {
      //std::cout << "deleting " << ht << " - " << wt << "\n";
      delete [] v; 
    }
  }

  TensorT(): shape(), v(nullptr), is_subtensor(false) {}
  TensorT(bool is_subtensor): shape(), v(nullptr), is_subtensor(is_subtensor) {};
  TensorT(const TensorT& other): shape(other.shape), v(new T[size]), is_subtensor(false) {
    std::copy(other.v, other.v+size, v);
  }

  TensorT(size_t size) : shape(size), v(nullptr), is_subtensor(false) { 
    v = new T[size];
    std::fill(v, v+size, 0); 
  }

  TensorT(Shape<dim> shape) : shape(shape), v(new T[size]), is_subtensor(false) { 
    std::fill(v, v+size, 0); 
  }

  TensorT(float *other_v, size_t size) : shape(size), v(nullptr), is_subtensor(false) { 
    v = new T[size];
    std::copy(other_v, other_v+size, v); 
  }

  T& operator[](int index) const {
    return v[index];
  }

  TensorT& operator*(float scaler) {
    for(size_t i = 0; i < size; i++) v[i] *= scaler;
    return *this;
  }

  void swap(TensorT& other) {
    shape.swap(other.shape);
    std::swap(other.v, v);
    std::swap(other.is_subtensor, is_subtensor);
  }

  TensorT& operator=(const TensorT& other) {
    TensorT temp = TensorT(other);
    swap(temp);
    return *this;
  }

  TensorT& operator=(TensorT& other) {
    other.is_subtensor;
    swap(other);
    return *this;
  }

  void set_is_subtensor(bool value) {
    is_subtensor = value;
  }

  private:
    static const size_t _dim = dim;
    bool is_subtensor;
};

#pragma GCC diagnostic pop

template <size_t dim>
using Tensor = TensorT<float, dim>;

typedef TensorT<float, 2> Matrix;
typedef TensorT<float, 1> Vector;

#endif
