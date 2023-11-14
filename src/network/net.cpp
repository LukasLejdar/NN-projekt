#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <filesystem>
#include <iterator>
#include <math.h>
#include <sched.h>
#include <string>
#include <iostream>
#include <cstdlib>
#include <algorithm>
#include <bits/stdc++.h>
#include <thread>
#include "activations.hpp"
#include "net.hpp"
#include "math.hpp"
#include "../mnist_reader.hpp"

Dense::Dense(const Dense& other): in_shape(other.in_shape), out_shape(other.out_shape) {
  w = other.w;
  b = other.b;
}

Dense::Dense(size_t in_shape, size_t out_shape): 
  in_shape(in_shape), 
  out_shape(out_shape),
  w(Matrix(out_shape, in_shape)),
  b(Matrix(out_shape, 1)) {}

void Dense::swap(Dense& other) {
    std::swap(in_shape, other.in_shape);
    std::swap(out_shape, other.out_shape);
    w.swap(other.w);
    b.swap(other.b);
  }

Dense& Dense::operator=(const Dense& other) {
  Dense temp(other);
  swap(temp);
  return *this;
}

// Net ----------------------------------------------

void Net::print_layer(size_t i, size_t t) {
  print_layer(i, threadscache[t]);
}

void Net::print_layer(size_t i, Cache& cache) {
    std::cout << "\nweights\n";
    printMat(layers[i].w);

    std::cout << "\nbiases\n";
    printMat(layers[i].b);

    std::cout << "\nact\n";
    printMat(cache.a[i]);

    std::cout << "\ndWeights\n";
    printMat(cache.dW[i]);

    std::cout << "\ndBiases\n";
    printMat(cache.dB[i]);

    std::cout << "\n";
}

void Net::initialize_cache(Cache& cache) {
  cache.b = new Matrix[layers_count];
  cache.w = new Matrix[layers_count];
  cache.dB = new Matrix[layers_count];
  cache.dW = new Matrix[layers_count];
  cache.a = new Matrix[layers_count+1] + 1; // leave space for input

  cache.a[-1] = Matrix(layers[0].in_shape, 1); // input
  cache.Y = Matrix(layers[layers_count-1].out_shape, 1);

  for(size_t i = 0; i < layers_count; i++) {
    cache.a[i] = Matrix(layers[i].out_shape, 1);
    cache.b[i] = Matrix(layers[i].out_shape, 1);
    cache.w[i] = Matrix(layers[i].out_shape, layers[i].in_shape);
    cache.dB[i] = Matrix(layers[i].out_shape, 1);
    cache.dW[i] = Matrix(layers[i].out_shape, layers[i].in_shape);
  }
}

Net::Net(Dense _layers[], size_t length) {
  layers_count = length;
  layers = new Dense[length];
  std::copy(_layers, _layers+length, layers);

  for(size_t t = 0; t < NTHREADS; t++) {
    initialize_cache(threadscache[t]);
  }

  // << "before randomize\n";
  for(size_t i = 0; i < length; i++) {
    randomizeMat(layers[i].w);
    auto [mean, varience] = getVarAndExp(layers[i].w);
    std::cout << "layer " << i << " weights mean " << mean << "\n";
    std::cout << "layer " << i << " weights varience " << varience << " expected " << 1.0 / layers[i].w.wt << "\n";
  }
}

//TODO: benchmark forwad_prop
Matrix& Net::forward_prop(Cache& cache) {
  for(size_t i = 0; i < layers_count; i++) {
    matMul<8>(layers[i].w, cache.a[i-1], cache.a[i]);
    addMat<8>(layers[i].b, cache.a[i]);
    
    if(i == layers_count-1) { softmax(cache.a[i].v, cache.a[i].ht); }
    else { relu(cache.a[i].v, cache.a[i].ht); }
  }

  return cache.a[layers_count - 1];
}

//TODO: benchmark backward_prop
void Net::back_prop(Cache& cache) {
  int i = layers_count-1;
  addMat<8, 1, -1>(cache.a[i], cache.Y, cache.dB[i]);
  mulMatAvT<8>(cache.dB[i], cache.a[i-1], cache.dW[i]);
  
  for(;i > 0;) {
    matMulATB<8>(layers[i].w, cache.dB[i], cache.dB[i-1]);
    i--;
    relu_backward(cache.dB[i].v, cache.a[i].v, cache.a[i].ht);
    mulMatAvT<8>(cache.dB[i], cache.a[i-1], cache.dW[i]);
  }
}

void Net::train(Cache* cache) {
  forward_prop(*cache);
  back_prop(*cache);
}

void Net::prepare_cache(Matrix& X, int y, Cache& cache) {
    copyMatricesOfSameSize(X, cache.a[-1]);
    zeroMat(cache.Y);
    cache.Y.v[y] = 1;
    
    for(size_t i = 0; i < layers_count; i++) {
      copyMatricesOfSameSize(layers[i].w, cache.w[i]);
      copyMatricesOfSameSize(layers[i].b, cache.b[i]);
    }
}
  
void Net::train_epochs(MnistReader& reader, int epochs) {
  std::thread* threads = new std::thread[NTHREADS];

  for(int e = 0; e < epochs; e++) {
    reader.loop_to_beg();
    
    float entr_sum = 0;
    for(int b = 0; b < reader.number_of_entries/NTHREADS; b++) {

      for(int t = 0; t < NTHREADS; t++) {
        reader.read_next();
        prepare_cache(reader.last_read, reader.last_lable, threadscache[t]);
        threads[t] = std::thread(&Net::train, this, &(threadscache[t]));
      }

      for (int t = 0; t < NTHREADS; t++) {
        threads[t].join();
        entr_sum += apply_gradient(threadscache[t]);
      }

      if(b % 50 == 49) {
        std::cout << "entropy " << b << " " << entr_sum/(NTHREADS*50) << " -------------------\n";
        entr_sum = 0;
      }
    }
  }
}

float Net::apply_gradient(Cache& cache) {
  for(size_t i = 0; i < layers_count; i++) {
    addMat<8>(cache.dW[i] * (-learning_rate), layers[i].w);
    addMat<8>(cache.dB[i] * (-learning_rate), layers[i].b);
  }

  return crossEntropy(cache.a[layers_count-1].v, cache.Y.v, cache.Y.ht);
}

void Net::test(MnistReader& reader) {
  Cache cache = threadscache[0];
  size_t out_shape = layers[layers_count -1].out_shape;
  Matrix preds = {out_shape, out_shape};

  float* count = new float[out_shape*out_shape];
  std::fill(count, count + out_shape*out_shape, 0);

  int total_correct = 0;

  for(int i = 0; i < reader.number_of_entries; i++) {
    reader.read_next();
    reader.last_read.ht = reader.last_read.ht*reader.last_read.wt;
    reader.last_read.wt = 1;

    Matrix preds = forward_prop(cache);
    auto max = std::distance(preds.v, std::max_element(preds.v, preds.v + preds.ht));

    count[reader.last_lable]++;
    preds[reader.last_lable][max]++; 
    if (max == reader.last_lable) total_correct++;
  }

  for(size_t x = 0; x < out_shape; x++) {
    for(size_t y = 0; y < out_shape; y++) {
      preds[x][y] = preds[x][y] / count[x];
    }
  }
  
  std::cout << "guess count\n";
  printMat(preds);
  std::cout << "drew\n";
  drawMat(preds, 1);

  std::cout << "\ntotal accuracy: " << static_cast<float>(total_correct) / static_cast<float>(reader.number_of_entries) << "\n";

  delete [] count;
}

