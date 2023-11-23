
#ifndef LAYER_H 
#define LAYER_H 

#include "math.hpp"

struct Dense {
  const Shape<2> in_shape, out_shape, w_shape;
  const Matrix w, maW, emaW; //moving averages
  const Matrix b, maB, emaB; 

  Dense(const Dense& other);
  Dense(size_t in, size_t out);
};

struct Convolutional {
  public:
    const Shape<4> k_shape;    
    const Shape<3> in_shape, out_shape;
    const Tensor<4> k, maK, emaK;
    const Tensor<3> b, maB, emaB;

    Convolutional(const Convolutional& other);
    Convolutional(Shape<3> in_shape, Shape<4> ker_shape);
};

struct Model {
  const size_t conv_count;
  const size_t dense_count;
  const size_t layers_count = conv_count + dense_count;

  const Convolutional* conv_layers;
  const Dense* dense_layers;

  Model(size_t conv_count, size_t dense_count, Convolutional* conv_layers, Dense* dense_layers);
};

struct ConvCache {
  size_t count;

  Tensor<3> *a, *dA; //activations a[-1] is a duplicate of trained sample 
  Tensor<3> *b, *dB;
  Tensor<4> *k, *dK;

  ~ConvCache() {
    delete [] (a-1); delete [] dA;
    delete [] b; delete [] dB;
    delete [] k; delete [] dK;
  }
};

struct DenseCache {
  size_t count;

  Matrix *a, *dA; // activations a[count] are predictions 
  Matrix *b, *dB;
  Matrix *w, *dW;

  ~DenseCache() {
    delete [] a; delete [] dA;
    delete [] b; delete [] dB;
    delete [] w; delete [] dW;
  }
};

struct Cache {
  size_t y; // corect label
  ConvCache conv;
  DenseCache dense;
};

void initialize_cache(Cache &cache, Model &model);

#endif
