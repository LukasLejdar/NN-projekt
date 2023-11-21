#include "layer.hpp"
#include "math.hpp"

Dense::Dense(const Dense& other): in_shape(other.in_shape), out_shape(other.out_shape) {
  w = other.w;
  b = other.b;
  emaW = other.emaW;
  emaB = other.emaB;
  maW = other.maW;
  maB = other.maB;
}

Dense::Dense(size_t in_shape, size_t out_shape): 
  in_shape(in_shape), 
  out_shape(out_shape),
  w(Matrix(out_shape, in_shape)),
  b(Matrix(out_shape, 1)),
  emaW(Matrix(out_shape, in_shape)),
  emaB(Matrix(out_shape, 1)),
  maW(Matrix(out_shape, in_shape)),
  maB(Matrix(out_shape, 1)) {}

//void Dense::swap(Dense& other) {
//    std::swap(in_shape, other.in_shape);
//    std::swap(out_shape, other.out_shape);
//    w.swap(other.w);
//    b.swap(other.b);
//  }
//
//Dense& Dense::operator=(const Dense& other) {
//  Dense temp(other);
//  swap(temp);
//  return *this;
//}

