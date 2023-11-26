
#ifndef LAYER_H 
#define LAYER_H 

#include "math.hpp"
#include <mutex>

struct Dense {
  const Shape<1> out_shape, in_shape;
  const Shape<2> w_shape;
  const Matrix w, maW, emaW; //moving averages
  const Vector b, maB, emaB; 

  Dense(const Dense& other);
  Dense(size_t in, size_t out);

  void randomize() const;
};

struct Convolutional {
  public:
    const Shape<4> k_shape;    
    const Shape<3> e_shape, out_shape, in_shape;
    const Shape<2> pooling;
    const Tensor<4> k, maK, emaK;
    const Tensor<3> b, maB, emaB;

    Convolutional(const Convolutional& other);
    Convolutional(Shape<3> in_shape, Shape<3> ker_shape, Shape<2> pooling);

    void randomize() const;
};

struct Model {
  const size_t conv_count;
  const size_t dense_count;
  const size_t layers_count = conv_count + dense_count;

  const Convolutional* conv_layers;
  const Dense* dense_layers;

  void randomize() const;

  Model(size_t conv_count, Convolutional* conv_layers, size_t dense_count, Dense* dense_layers);
};

struct DenseCache {
  size_t count;

  Vector *a, *dA;
  Vector *b, *dB;
  Matrix *w, *dW;

  ~DenseCache() {
    delete [] (a-1); delete [] (dA-1);
    delete [] b; delete [] dB;
    delete [] w; delete [] dW;
  }
};

struct ConvCache {
  size_t count;

  Tensor<3> *a, *dA;
  Tensor<3> *b, *dB;
  Tensor<4> *k, *dK;
  Tensor<3> *out, *dOut;
  Shape<2>* pooling;
  TensorT<size_t, 3> *loc;

  ~ConvCache() {
    delete [] a; delete [] dA;
    delete [] b; delete [] dB;
    delete [] k; delete [] dK;
    delete [] (out-1); delete [] dOut;
    delete [] pooling; delete [] loc;
  }
};

struct Cache {
  size_t y; // corect label
  float accuracy;
  ConvCache conv;
  DenseCache dense;
};

void initialize_cache(Cache &cache, Model &model);

void drawConv(Cache& cache);

#endif
