#include "layer.hpp"

Dense::Dense(const Dense& other): 
  in_shape(other.in_shape), 
  out_shape(other.out_shape),
  w(other.w),
  maW(other.maW),
  emaW(other.emaW),
  b(other.b),
  maB(other.maB),
  emaB(other.emaB)
{}

Dense::Dense(size_t in_shape, size_t out_shape): 
  in_shape(in_shape), 
  out_shape(out_shape),
  w(Matrix(out_shape, in_shape)),
  b(Matrix(out_shape, 1)),
  emaW(Matrix(out_shape, in_shape)),
  emaB(Matrix(out_shape, 1)),
  maW(Matrix(out_shape, in_shape)),
  maB(Matrix(out_shape, 1)) 
{}

Convolutional::Convolutional(const Convolutional& other): 
  in_shape(other.in_shape), 
  out_shape(other.out_shape),
  k(other.k),
  maK(other.maK),
  emaK(other.emaK),
  b(other.b),
  maB(other.maB),
  emaB(other.emaB)
{}

Convolutional::Convolutional(Shape<3> in_shape, Shape<4> ker_shape):
  in_shape(in_shape),
  out_shape(ker_shape[0], in_shape[1], in_shape[2]),
  k(ker_shape),
  maK(ker_shape),
  emaK(ker_shape),
  b(out_shape),
  maB(out_shape),
  emaB(out_shape)
{}
