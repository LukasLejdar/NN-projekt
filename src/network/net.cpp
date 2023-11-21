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

void initialize_cache(Cache& cache, Dense layers[], size_t layers_count) {
  cache.layers_count = layers_count;
  cache.b = new Matrix[layers_count];
  cache.w = new Matrix[layers_count];
  cache.a = new Matrix[layers_count+2] + 1; // leave space for input
  cache.dA = new Matrix[layers_count];
  cache.dB = new Matrix[layers_count];
  cache.dW = new Matrix[layers_count];

  cache.a[-1] = Matrix(layers[0].in_shape, 1); // input

  for(size_t i = 0; i < layers_count; i++) {
    cache.a[i] = Matrix(layers[i].out_shape, 1);
    cache.b[i] = Matrix(layers[i].out_shape, 1);
    cache.w[i] = Matrix(layers[i].out_shape, layers[i].in_shape);
    cache.dA[i] = Matrix(layers[i].out_shape, 1);
    cache.dB[i] = Matrix(layers[i].out_shape, 1);
    cache.dW[i] = Matrix(layers[i].out_shape, layers[i].in_shape);
  }
}

Net::Net(Dense _layers[], size_t length) {
  layers_count = length;
  layers = new Dense[length];
  std::copy(_layers, _layers+length, layers);

  mtx = new std::mutex[length];

  for(size_t t = 0; t < NTHREADS; t++) {
    initialize_cache(threadscache[t], _layers, layers_count);
  }

  for(size_t i = 0; i < length; i++) { randomizeMat(layers[i].w); }
}

//TODO: benchmark forwad_prop
Matrix& forward_prop(Cache& cache) {
  for(size_t i = 0; i < cache.layers_count; i++) {
    matMul<8>(cache.w[i], cache.a[i-1], cache.a[i]);
    addMat<8>(cache.b[i], cache.a[i]);

    if(i != cache.layers_count-1) relu(cache.a[i].v, cache.a[i].ht);
    else softmax(cache.a[i].v, cache.a[i].ht);
  }
  
  return cache.a[cache.layers_count - 1];
}

//TODO: benchmark backward_prop
void back_prop(Cache& cache) {
  int i = cache.layers_count-1;
  copyMatricesOfSameSize(cache.a[i], cache.dA[i]);
  cache.dA[i].v[cache.y] -= 1;

  addMat<8>(cache.dA[i], cache.dB[i]);
  mulMatAvT<8, false>(cache.dA[i], cache.a[i-1], cache.dW[i]);
  
  for(;i > 0;) {
    matMulATB<8>(cache.w[i], cache.dA[i], cache.dA[i-1]);
    i--;
    relu_backward(cache.dA[i].v, cache.a[i].v, cache.a[i].ht);
    addMat<8>(cache.dA[i], cache.dB[i]);
    mulMatAvT<8, false>(cache.dA[i], cache.a[i-1], cache.dW[i]);
  }
}

void Net::apply_gradient(Cache& cache, size_t t) {
  //for(size_t i = 0; i < layers_count; i++) {
  //  cache.dW[i] * (-learning_rate / (1+decay_rate*epoch));
  //  cache.dB[i] * (-learning_rate / (1+decay_rate*epoch));
  //}

  for(size_t i = 0; i < layers_count; i++) {
    mtx[i].lock();
    adam(cache.dW[i], layers[i].emaW, layers[i].maW, learning_rate, decay_rate1, decay_rate2, t);
    adam(cache.dB[i], layers[i].emaB, layers[i].maB, learning_rate, decay_rate1, decay_rate2, t);

    addMat<8>(cache.dW[i], layers[i].w);
    addMat<8>(cache.dB[i], layers[i].b);
    mtx[i].unlock();
  }
}

void zeroDerivatives(Cache& cache) {
  for(size_t i = 0; i < cache.layers_count; i++) {
    zeroMat(cache.dW[i]);
    zeroMat(cache.dB[i]);
  }
}

void Net::prepare_cache(Matrix& X, int y, Cache& cache) {
    copyMatricesOfSameSize(X, cache.a[-1]);
    cache.y = y;
    
    for(size_t i = 0; i < layers_count; i++) {
      mtx[i].lock();
      copyMatricesOfSameSize(layers[i].w, cache.w[i]);
      copyMatricesOfSameSize(layers[i].b, cache.b[i]);
      mtx[i].unlock();
    }
}

void Net::train(Cache* cache, MnistReader* reader, int t_index, int epoch) {
  int period = 5000;
  float entrsum = 0;
  zeroDerivatives(*cache);

  while(true) {
    if (!reader->read_next()) return; 

    prepare_cache(reader->last_read, reader->last_lable, *cache);
    forward_prop(*cache);
    back_prop(*cache);

    if(reader->index % mini_batch == 0) {
      apply_gradient(*cache, reader->number_of_entries*NTHREADS*epoch + reader->index*NTHREADS); 
      zeroDerivatives(*cache);
    }

    entrsum += crossEntropy(cache->a[layers_count-1].v, cache->y);
    if(entrsum != entrsum) {
      std::cout << reader->index << "entr sum is nan\n"; 
      entrsum = 0;
    }
    if(reader->index % period == 0) {
      if(reader->index % (period*NTHREADS) == period*t_index) {
        
        std::cout << " executing sample " << reader->index << " on thread " << t_index << " entropy average " << entrsum / period << "\n";
      }
      entrsum = 0;
    }
  }
}

void Net::train_epochs(MnistReader& reader, int epochs, MnistReader& test_reader) {
  std::thread threads[NTHREADS-1];

  size_t control_size = reader.number_of_entries / 6;
  size_t sample_size = reader.number_of_entries - control_size;

  MnistReader control_data = MnistReader(reader, sample_size, reader.number_of_entries);
  MnistReader sample_data = MnistReader(reader, 0, 10000);

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
      std::cout << "\nresults for sample data ########################" << "\n";
      test(sample_data); 
      std::cout << "\nresults for control data ########################" << "\n";
      float accuracy = test(control_data); 
      if(accuracy >= 0.894) {
        std::cout << "\nresults for test data %%%%%%%%%%%%%%%%%%%%%%%%%%%" << "\n";
        test(test_reader);
      }
    }

  }
  
  for(auto th_reader : threads_readers) delete th_reader;
  std::cout << "training finished" << "\n"; 
}

float Net::test(MnistReader& reader) {
  reader.loop_to_beg();

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

  float percentage = static_cast<float>(total_correct) / reader.number_of_entries;
  
  std::cout << "\ntotal accuracy: " << percentage << "\n";
  std::cout << "guess count\n";
  printMat(results);
  drawMat(results, 1);
  return percentage;
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


