
#ifndef LAYER_H 
#define LAYER_H 

#include "math.hpp"

struct Dense {
  size_t in_shape, out_shape;
  Matrix w;
  Matrix b;

  Dense(): in_shape(0), out_shape(0), w({}), b({}) {}
  Dense(const Dense& other);
  Dense(size_t in_shape, size_t out_shape);
  void swap(Dense& other);
  Dense& operator=(const Dense& other);
};

#endif