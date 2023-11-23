#include "layer.hpp"

Dense::Dense(const Dense& other): 
  in_shape(other.in_shape), 
  out_shape(other.out_shape),
  w_shape(other.w_shape),
  w(other.w),
  maW(other.maW),
  emaW(other.emaW),
  b(other.b),
  maB(other.maB),
  emaB(other.emaB)
{}

Dense::Dense(size_t in, size_t out): 
  in_shape(in, 1), 
  out_shape(out, 1),
  w_shape(out, in),
  w(Matrix(w_shape)),
  maW(Matrix(w_shape)),
  emaW(Matrix(w_shape)),
  b(Matrix(out_shape)),
  maB(Matrix(out_shape)),
  emaB(Matrix(out_shape))
{}

Convolutional::Convolutional(const Convolutional& other): 
  ker_shape(other.ker_shape),
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
  ker_shape(ker_shape),
  in_shape(in_shape),
  out_shape(ker_shape[0], in_shape[1], in_shape[2]),
  k(ker_shape),
  maK(ker_shape),
  emaK(ker_shape),
  b(out_shape),
  maB(out_shape),
  emaB(out_shape)
{}

Model::Model(size_t conv_count, size_t dense_count, Convolutional* conv_layers, Dense* dense_layers) :
  conv_count(conv_count), 
  dense_count(dense_count), 
  conv_layers(conv_layers), 
  dense_layers(dense_layers) {}


void initialize_cache(Cache& cache, Model& model) {
  cache.conv.count = model.conv_count;
  cache.conv.a = new Tensor<3>[model.conv_count+1]+1; // leave space for output
  cache.conv.dA = new Tensor<3>[model.conv_count];
  cache.conv.b = new Tensor<3>[model.conv_count];
  cache.conv.dB = new Tensor<3>[model.conv_count];
  cache.conv.k = new Tensor<4>[model.conv_count];
  cache.conv.dK = new Tensor<4>[model.conv_count];

  cache.dense.count = model.dense_count;
  cache.dense.a = new Matrix[model.dense_count+1]; // leave space for preds 
  cache.dense.dA = new Matrix[model.dense_count];
  cache.dense.b = new Matrix[model.dense_count]; 
  cache.dense.dB = new Matrix[model.dense_count];
  cache.dense.w = new Matrix[model.dense_count]; 
  cache.dense.dW = new Matrix[model.dense_count];

  cache.conv.a[-1] = Tensor<3>(model.conv_layers[0].in_shape); // input
  cache.dense.a[model.dense_count] = Matrix(model.dense_layers[model.dense_count].out_shape); // predictions

  for(size_t i = 0; i < model.conv_count; i++) {
    cache.conv.a[i] = Tensor<3>(model.conv_layers[i].in_shape); 
    cache.conv.dA[i] = Tensor<3>(model.conv_layers[i].in_shape);
    cache.conv.b[i] = Tensor<3>(model.conv_layers[i].out_shape); 
    cache.conv.dB[i] = Tensor<3>(model.conv_layers[i].out_shape);
    cache.conv.k[i] = Tensor<4>(model.conv_layers[i].k_shape); 
    cache.conv.dK[i] = Tensor<4>(model.conv_layers[i].k_shape);
  }

  for(size_t i = 0; i < model.dense_count; i++) {
    cache.dense.a[i] = Matrix(model.dense_layers[i].in_shape); 
    cache.dense.dA[i] = Matrix(model.dense_layers[i].in_shape);
    cache.dense.b[i] = Matrix(model.dense_layers[i].out_shape); 
    cache.dense.dB[i] = Matrix(model.dense_layers[i].out_shape);
    cache.dense.w[i] = Matrix(model.dense_layers[i].w_shape); 
    cache.dense.dW[i] = Matrix(model.dense_layers[i].w_shape);
  }
}
