#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <filesystem>
#include <functional>
#include <iterator>
#include <math.h>
#include <mutex>
#include <sched.h>
#include <string>
#include <iostream>
#include <cstdlib>
#include <algorithm>
#include <bits/stdc++.h>
#include <thread>
#include <vector>
#include "activations.hpp"
#include "net.hpp"
#include "math.hpp"
#include "../mnist_reader.hpp"

void Net::initialize_cache(Cache& cache) {
  cache.layers_count = layers_count;
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
  mtx = new std::mutex[length];
  layers = new Dense[length];
  std::copy(_layers, _layers+length, layers);

  for(size_t t = 0; t < NTHREADS; t++) {
    initialize_cache(threadscache[t]);
  }

  for(size_t i = 0; i < length; i++) { randomizeMat(layers[i].w); }
}

//TODO: benchmark forwad_prop
Matrix& forward_prop(Cache& cache) {
  for(size_t i = 0; i < cache.layers_count; i++) {
    matMul<8>(cache.w[i], cache.a[i-1], cache.a[i]);
    addMat<8>(cache.b[i], cache.a[i]);
    
    if(i == cache.layers_count-1) { softmax(cache.a[i].v, cache.a[i].ht); }
    else { relu(cache.a[i].v, cache.a[i].ht); }
  }

  return cache.a[cache.layers_count - 1];
}

//TODO: benchmark backward_prop
void back_prop(Cache& cache) {
  int i = cache.layers_count-1;
  addMat<8, 1, -1>(cache.a[i], cache.Y, cache.dB[i]);
  mulMatAvT<8>(cache.dB[i], cache.a[i-1], cache.dW[i]);
  
  for(;i > 0;) {
    matMulATB<8>(cache.w[i], cache.dB[i], cache.dB[i-1]);
    i--;
    relu_backward(cache.dB[i].v, cache.a[i].v, cache.a[i].ht);
    mulMatAvT<8>(cache.dB[i], cache.a[i-1], cache.dW[i]);
  }
}

void Net::apply_gradient(Cache& cache, int epoch, int sample) {
  for(size_t i = 0; i < layers_count; i++) {
    cache.dW[i] * (-learning_rate / (1+decay_rate*epoch));
    cache.dB[i] * (-learning_rate / (1+decay_rate*epoch));
  }

  for(size_t i = 0; i < layers_count; i++) {
    mtx[i].lock();
    addMat<8>(cache.dW[i], layers[i].w);
    addMat<8>(cache.dB[i], layers[i].b);
    mtx[i].unlock();
  }
}

void Net::prepare_cache(Matrix& X, int y, Cache& cache) {
    copyMatricesOfSameSize(X, cache.a[-1]);
    zeroMat(cache.Y);
    cache.Y.v[y] = 1;
    
    for(size_t i = 0; i < layers_count; i++) {
      //mtx[i].lock();
      copyMatricesOfSameSize(layers[i].w, cache.w[i]);
      copyMatricesOfSameSize(layers[i].b, cache.b[i]);
      //mtx[i].unlock();
    }
}

void Net::train(Cache* cache, MnistReader* reader, int t_index, int epoch) {
  int period = 1000;
  float entrsum = 0;
  while(true) {
    if (!reader->read_next()) return;

    prepare_cache(reader->last_read, reader->last_lable, *cache);
    forward_prop(*cache);
    back_prop(*cache);
    apply_gradient(*cache, epoch, reader->index); 

    entrsum += crossEntropy(cache->a[layers_count-1].v, cache->Y.v, cache->Y.ht);
    if(reader->index % period == 0) {
      if(reader->index % (period*NTHREADS) == period*t_index) {
        std::cout << "executing sample " << reader->index << " on thread " << t_index << " entropy average " << entrsum / period << "\n";
      }
      entrsum = 0;
    }
  }
}

void Net::train_epochs(MnistReader& reader, int epochs) {
  std::thread threads[NTHREADS-1];

  size_t control_size = reader.number_of_entries / 6;
  size_t sample_size = reader.number_of_entries - control_size;

  MnistReader control_data = MnistReader(reader, sample_size, reader.number_of_entries);

  //every thread has its own reader
  size_t threads_lot = sample_size / NTHREADS;
  MnistReader* threads_readers[NTHREADS];
  for(int t = 0; t < NTHREADS; t++) {
    threads_readers[t] = new MnistReader(reader, t*threads_lot, (t+1)*threads_lot);
  }

  for(int e = 0; e < epochs; e++) {
    std::cout << "epoch " << e << "\n"; 
    for(int t = 0; t < NTHREADS-1; t++) 
      threads[t] = std::thread(&Net::train, this, &threadscache[t], threads_readers[t], t, e);

    train(&threadscache[NTHREADS-1], threads_readers[NTHREADS-1], NTHREADS-1, e);
    threads_readers[NTHREADS-1]->loop_to_beg();

    for(int t = 0; t < NTHREADS-1; t++) { 
      threads[t].join();
      threads_readers[t]->loop_to_beg();
    }

    if(e % 5 == 0) { 
      control_data.loop_to_beg();
      test(control_data); 
    }
  }
  
  for(auto th_reader : threads_readers) delete th_reader;
  std::cout << "training finished" << "\n"; 
}

void Net::test(MnistReader& reader) {
  Cache& cache = threadscache[0];
  size_t out_shape = layers[layers_count -1].out_shape;
  Matrix results = {out_shape, out_shape};
  MatrixT<int> count = {out_shape, 1};

  int total_correct = 0;

  for(int i = 0; i < reader.number_of_entries; i++) {
    reader.read_next();
    prepare_cache(reader.last_read, reader.last_lable, cache);

    Matrix preds = forward_prop(cache);
    int max = std::distance(preds.v, std::max_element(preds.v, preds.v + preds.ht));

    count[reader.last_lable][0]++;
    results[reader.last_lable][max]++; 
    if (max == reader.last_lable) total_correct++;
  }

  for(size_t x = 0; x < out_shape; x++) {
    for(size_t y = 0; y < out_shape; y++) {
      results[x][y] = results[x][y] / count[x][0];
    }
  }
  
  std::cout << "\ntotal accuracy: " << static_cast<float>(total_correct) / static_cast<float>(reader.number_of_entries) << "\n";
  std::cout << "guess count\n";
  printMat(results);
  drawMat(results, 1);
}

void Net::print_layer(size_t i, Cache& cache) {    
  std::cout << "layer " << i << ":\n";
  std::cout << "\nactivations\n";
  printMat(cache.a[i]);
  std::cout << "\nweights\n";
  printMat(layers[i].w);
  std::cout << "\nbiases\n";
  printMat(layers[i].b);
  std::cout << "\ndW\n";
  printMat(cache.dW[i]);
  std::cout << "\ndB\n";
  printMat(cache.dB[i]);
  std::cout << "\n";
}

