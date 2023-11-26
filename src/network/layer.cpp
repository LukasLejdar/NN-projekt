#include "layer.hpp"
#include "math.hpp"
#include <cassert>
#include <string>

Dense::Dense(const Dense& other): 
  out_shape(other.out_shape),
  in_shape(other.in_shape), 
  w_shape(other.w_shape),
  w(other.w),
  maW(other.maW),
  emaW(other.emaW),
  b(other.b),
  maB(other.maB),
  emaB(other.emaB)
{}

Dense::Dense(size_t in, size_t out): 
  out_shape(out),
  in_shape(in), 
  w_shape(out, in),
  w(Matrix(w_shape)),
  maW(Matrix(w_shape)),
  emaW(Matrix(w_shape)),
  b(Vector(out_shape)),
  maB(Vector(out_shape)),
  emaB(Vector(out_shape))
{}

Convolutional::Convolutional(const Convolutional& other): 
  k_shape(other.k_shape),
  e_shape(other.e_shape),
  in_shape(other.in_shape), 
  pooling(other.pooling),
  k(other.k),
  maK(other.maK),
  emaK(other.emaK),
  b(other.b),
  maB(other.maB),
  emaB(other.emaB)
{}

Convolutional::Convolutional(Shape<3> in_shape, Shape<3> k_size, Shape<2> pooling):
  k_shape(k_size[0], in_shape[0], k_size[1], k_size[2]),
  e_shape(k_size[0], in_shape.ht - k_size.ht +1, in_shape.wt - k_size.wt +1),
  out_shape(e_shape[0], e_shape.ht/pooling.ht, e_shape.wt/pooling.wt),
  in_shape(in_shape),
  pooling(pooling),
  k(k_shape),
  maK(k_shape),
  emaK(k_shape),
  b(e_shape),
  maB(e_shape),
  emaB(e_shape)
{}

Model::Model(size_t conv_count, Convolutional* conv_layers, size_t dense_count, Dense* dense_layers) :
  conv_count(conv_count), 
  dense_count(dense_count), 
  conv_layers(conv_layers), 
  dense_layers(dense_layers) {
    for(size_t i = 0; i < conv_count-1; i++) {
      assert(conv_layers[i].out_shape.ht == conv_layers[i].e_shape.ht / conv_layers[i].pooling.ht );
      assert(conv_layers[i].out_shape.wt == conv_layers[i].e_shape.wt / conv_layers[i].pooling.wt );
      assert(conv_layers[i].out_shape == conv_layers[i+1].in_shape);
    }

    assert(conv_layers[conv_count-1].out_shape.ht == conv_layers[conv_count-1].e_shape.ht / conv_layers[conv_count-1].pooling.ht );
    assert(conv_layers[conv_count-1].out_shape.wt == conv_layers[conv_count-1].e_shape.wt / conv_layers[conv_count-1].pooling.wt );
    assert(conv_layers[conv_count-1].out_shape.size == dense_layers[0].in_shape.size);

    for(size_t i = 0; i < dense_count-1; i++) {
      assert(dense_layers[i].out_shape == dense_layers[i+1].in_shape);
    }
}


void initialize_cache(Cache& cache, Model& model) {
  cache.conv.count = model.conv_count;
  cache.conv.a = new Tensor<3>[model.conv_count];
  cache.conv.dA = new Tensor<3>[model.conv_count];
  cache.conv.b = new Tensor<3>[model.conv_count];
  cache.conv.dB = new Tensor<3>[model.conv_count];
  cache.conv.k = new Tensor<4>[model.conv_count];
  cache.conv.dK = new Tensor<4>[model.conv_count];
  cache.conv.out = new Tensor<3>[model.conv_count+1]+1; // out[-1] is a copy of input
  cache.conv.dOut = new Tensor<3>[model.conv_count];
  cache.conv.pooling = new Shape<2>[model.conv_count];
  cache.conv.loc = new TensorT<size_t, 3>[model.conv_count];

  cache.dense.count = model.dense_count;
  cache.dense.a = new Vector[model.dense_count+1]+1; // a[-1] is reshaped output of conv (dense.a[-1].v points to conv.out[count-1].v)
  cache.dense.dA = new Vector[model.dense_count+1]+1; // similarly dA[-1].v is conv.dOut[count-1].v
  cache.dense.b = new Vector[model.dense_count]; 
  cache.dense.dB = new Vector[model.dense_count];
  cache.dense.w = new Matrix[model.dense_count]; 
  cache.dense.dW = new Matrix[model.dense_count];

  for(size_t i = 0; i < model.conv_count; i++) {
    cache.conv.a[i] = Tensor<3>(model.conv_layers[i].e_shape); 
    cache.conv.dA[i] = Tensor<3>(model.conv_layers[i].e_shape);
    cache.conv.b[i] = Tensor<3>(model.conv_layers[i].e_shape); 
    cache.conv.dB[i] = Tensor<3>(model.conv_layers[i].e_shape);
    cache.conv.k[i] = Tensor<4>(model.conv_layers[i].k_shape); 
    cache.conv.dK[i] = Tensor<4>(model.conv_layers[i].k_shape);
    cache.conv.out[i] = Tensor<3>(model.conv_layers[i].out_shape);
    cache.conv.dOut[i] = Tensor<3>(model.conv_layers[i].out_shape);
    cache.conv.pooling[i] = Shape<2>(model.conv_layers[i].pooling);
    cache.conv.loc[i] = TensorT<size_t, 3>(model.conv_layers[i].out_shape);
  }

  for(size_t i = 0; i < model.dense_count; i++) {
    cache.dense.a[i] = Vector(model.dense_layers[i].out_shape); 
    cache.dense.dA[i] = Vector(model.dense_layers[i].out_shape);
    cache.dense.b[i] = Vector(model.dense_layers[i].out_shape); 
    cache.dense.dB[i] = Vector(model.dense_layers[i].out_shape);
    cache.dense.w[i] = Matrix(model.dense_layers[i].w_shape); 
    cache.dense.dW[i] = Matrix(model.dense_layers[i].w_shape);
  }

  cache.conv.out[-1] = Tensor<3>(model.conv_layers[0].in_shape); // input
  cache.conv.out[cache.conv.count-1].vectorize(cache.dense.a[-1]);
  cache.conv.dOut[cache.conv.count-1].vectorize(cache.dense.dA[-1]);
}

void Model::randomize() const {
  for(size_t i = 0; i < dense_count; i++) dense_layers[i].randomize();
  for(size_t i = 0; i < conv_count; i++) conv_layers[i].randomize();
}

void Dense::randomize() const { 
  ::randomize(w, 0, 2.0/(w.ht + w.wt));
};

void Convolutional::randomize() const { 
  ::randomize(k, 0, 2.0/(k.ht + k.wt));
}

void drawConv(Cache& cache) {
  std::cout << "\nout -1" << "\n";
  draw3D(cache.conv.out[-1]);

  for(size_t i = 0; i < cache.conv.count; i++) {
    std::cout << "\nk " << i << "\n";
    drawKernels(cache.conv.k[i]);

    std::cout << "\ndK " << i << "\n";
    drawKernels(cache.conv.dK[i]);

    std::cout << "\nb " << i << "\n";
    draw3D(cache.conv.b[i]);

    std::cout << "\ndB " << i << "\n";
    draw3D(cache.conv.dB[i]);

    std::cout << "\na " << i << "\n";
    draw3D(cache.conv.a[i]);

    std::cout << "\ndA " << i << "\n";
    draw3D(cache.conv.dA[i]);

    std::cout << "\nout " << i << "\n";
    draw3D(cache.conv.out[i]);

    std::cout << "\ndOut " << i << "\n";
    draw3D(cache.conv.dOut[i]);
  }
}

