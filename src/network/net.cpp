#include <cstdio>
#include <cstring>
#include <algorithm>
#include <iterator>
#include <string>
#include <iostream>
#include <cstdlib>
#include <bits/stdc++.h>
#include "net.hpp"
#include "math.hpp"

// activations --------------------------------------

void relu(float v[], int length) {
  for(int i = 0; i < length; i++) {
    v[i]=std::max<float>(0.0, v[i]);
  }
}

void relu_backward(float dA[], float A[], int length, float dZ[]) {
  for(int i = 0; i < length; i++) {
    dZ[i] = (A[i] > 1) * dA[i];
  }
}

void sigmoid(float v[], int length) {
  for(int i = 0; i < length; i++) {
    v[i] = 1 / (1 + exp(-v[i]));
  }
}

void softmax(float v[], int length) {
  float sum = 0;
  for(int i = 0; i < length; i++) {
    v[i] = exp(v[i]);
    sum += v[i];
  }
  for(int i = 0; i < length; i++) {
    v[i] = v[i] / sum;
  }
}

// Net ----------------------------------------------

void initialize_layer(Dense &dense) {
  float *weights = new float[dense.in_shape*dense.out_shape];
  float *biases = new float[dense.out_shape];

  dense.w.v = weights; 
  dense.w.ht = dense.out_shape; 
  dense.w.wt = dense.in_shape;

  dense.b.v = biases; 
  dense.b.ht = dense.out_shape; 
  dense.b.wt = 1;

  randomizeMat(dense.w);
  randomizeMat(dense.b);

  switch (dense.acti) {
    case RELU: 
      dense.activation = relu; 
      dense.back_activation = relu_backward;
      break;
    case SIGMOID: dense.activation = sigmoid; break;
    case SOFTMAX: dense.activation = softmax; break;
    default: dense.activation = relu; break;
  }

}

void initialize_cache(DenseCache &cache) {
  float *dWeights = new float[cache.in_shape*cache.out_shape];
  float *dBiases = new float[cache.out_shape];
  float *activations = new float[cache.out_shape];

  memset(dWeights, 0, cache.in_shape*cache.out_shape*sizeof(float));
  memset(dBiases, 0, cache.out_shape*sizeof(float));
  memset(activations, 0,cache.out_shape*sizeof(float));

  cache.dW.v = dWeights; 
  cache.dW.ht = cache.out_shape; 
  cache.dW.wt = cache.in_shape;

  cache.dB.v = dBiases; 
  cache.dB.ht = cache.out_shape; 
  cache.dB.wt = 1;

  cache.a.v = activations;
  cache.a.ht = cache.out_shape;
  cache.a.wt = 1;
}

Net::Net(Dense layers[], int length) {
  layers_count = length;
  this->layers = layers;

  assert(layers[length-1].acti == SOFTMAX);

  for(int i = 0; i < NTHREADS; i++) {
    DenseCache* cache{new DenseCache[length]};
    threadscache[i] = cache;
  }

  for(int i = 0; i < length; i++) {
    initialize_layer(layers[i]);
    for(int t = 0; t < NTHREADS; t++) {
      threadscache[t][i].in_shape = layers[i].in_shape;
      threadscache[t][i].out_shape = layers[i].out_shape;
      initialize_cache(threadscache[t][i]);
    }
  }

  for(int i = 1; i < length; i++) {
    for(int t = 0; t < NTHREADS; t++) {
      threadscache[t][i].a_prev = &(threadscache[t][i-1].a);
    }
  }
}

void Net::forward_prop(Matrix& X, int thread_i) {
  DenseCache* caches = threadscache[thread_i]; // activations
  caches[0].a_prev = &X;
  
  for(int i = 0; i < layers_count; i++) {
    mulMat(layers[i].w, *(caches[i].a_prev), caches[i].a);
    addMat(layers[i].b, caches[i].a, caches[i].a);
    layers[i].activation(caches[i].a.v, caches[i].a.ht);
  }

  for(int i = 0; i < layers_count; i++ ) {
    std::cout << "\nweights\n";
    printMat(layers[i].w);

    std::cout << "\nbiases\n";
    printMat(layers[i].b);

    std::cout << "\nact\n";
    printMat(caches[i].a);
  
  }
}

void Net::train(Matrix& X, int y, int thread_i) {
  float Yv[layers[layers_count-1].out_shape];
  Matrix Y = {layers[layers_count-1].out_shape, 1, Yv};
  memset(Y.v, 0, layers[layers_count-1].out_shape*sizeof(float));
  Y.v[y] = 1;

  forward_prop(X, thread_i);
  backward_prop(Y, thread_i);

  for(int i = 0; i < layers_count; i++) {
    addMat(layers[i].w, threadscache[thread_i][i].dW * -0.05, layers[i].w);
    addMat(layers[i].b, threadscache[thread_i][i].dB * -0.05, layers[i].b);
  }
}

void transposeColORowMat(Matrix& rowOrColMat) {
  int ht = rowOrColMat.ht;
  rowOrColMat.ht = rowOrColMat.wt;
  rowOrColMat.wt = ht;
}

void Net::backward_prop(Matrix &Y, int thread_i) {
  DenseCache* caches = threadscache[thread_i];

  int i = layers_count-1;
  addMat(caches[i].a, Y*-1, caches[i].dB); //cross entropy softmax, derivation, dB = DZ 
  mulMatABT(caches[i].dB, *(caches[i].a_prev), caches[i].dW);
  
  for(;i >= 1;) {
    float vdA[caches[i].a_prev->ht];
    Matrix dA_prev = {caches[i].a_prev->ht, 1,vdA};
    mulMatATB(layers[i].w, caches[i].dB, dA_prev);
    i--;
    layers[i].back_activation(vdA, caches[i].a.v, caches[i].a.ht, caches[i].dB.v); // dB = dZ
    mulMatABT(caches[i].dB, *(caches[i].a_prev), caches[i].dW);
  }

  for(int i = layers_count-1; i >= 0; i--) {
    std::cout << "\ndBiases\n";
    printMat(caches[i].dB);

    std::cout << "\ndWeights\n";
    printMat(caches[i].dW);
  }
}

