
#ifndef LAYER_H 
#define LAYER_H 

#include "math.hpp"

struct Dense {
  const size_t in_shape, out_shape;
  const Matrix w, maW, emaW; //moving averages
  const Matrix b, maB, emaB; 

  Dense(const Dense& other);
  Dense(size_t in_shape, size_t out_shape);
};

struct Convolutional {
  public:
    const Shape<3> in_shape;
    const Shape<3> out_shape;    
    const Tensor<4> k, maK, emaK;
    const Tensor<3> b, maB, emaB;

    Convolutional(const Convolutional& other);
    Convolutional(Shape<3> in_shape, Shape<4> ker_shape);
};

#endif
