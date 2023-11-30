#include <array>
#include <cassert>
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
#include <thread>
#include <tuple>
#include <utility>
#include <vector>
#include "activations.hpp"
#include "layer.hpp"
#include "net.hpp"
#include "math.hpp"
#include "../mnist_reader.hpp"

Net::Net(Model& model): model(model) {
  conv_mtx = new std::mutex[model.conv_count];
  dense_mtx = new std::mutex[model.dense_count];
  for(size_t t = 0; t < NTHREADS; t++) initialize_cache(threadscache[t], model);
  model.randomize();
}

Vector& forward_prop(Cache& cache) {
  assert(cache.conv.out[cache.conv.count-1].v == cache.dense.a[-1].v);

  for(size_t i = 0; i < cache.conv.count; i++) {
    correlateAv(cache.conv.k[i], cache.conv.out[i-1], cache.conv.a[i]);
    addTens(cache.conv.b[i], cache.conv.a[i]);
    relu(cache.conv.a[i]);
    maxPooling(cache.conv.a[i], cache.conv.pooling[i], cache.conv.out[i], cache.conv.loc[i]);
  }
  
  for(size_t i = 0; i < cache.dense.count; i++) {
    matMulAv(cache.dense.w[i], cache.dense.a[i-1], cache.dense.a[i]);
    addTens(cache.dense.b[i], cache.dense.a[i]);
    if(i != cache.dense.count -1) relu(cache.dense.a[i]);
  }

  softmax(cache.dense.a[cache.dense.count-1]);
  return cache.dense.a[cache.dense.count -1];
}

void back_prop(Cache& cache) {
  assert(cache.dense.dA[-1].v == cache.conv.dOut[cache.conv.count-1].v );

  int i = cache.dense.count-1;
  copyToTensorOfSameSize(cache.dense.a[i], cache.dense.dA[i]);
  cache.dense.dA[i].v[cache.y] -= 1;

  for(;i >= 0; i--) {
    addTens(cache.dense.dA[i], cache.dense.dB[i]);
    matMulvvT<false>(cache.dense.dA[i], cache.dense.a[i-1], cache.dense.dW[i]);
    matMulATv(cache.dense.w[i], cache.dense.dA[i], cache.dense.dA[i-1]);
    if (i != 0) relu_backward(cache.dense.dA[i-1], cache.dense.a[i-1]);
  }

  for(i = cache.conv.count-1; i >= 0; i--) {
    maxPooling_backward(cache.conv.dOut[i], cache.conv.dA[i], cache.conv.loc[i]);
    relu_backward(cache.conv.dA[i], cache.conv.a[i]);
    addTens(cache.conv.dA[i], cache.conv.dB[i]);
    correlatevvT<false>(cache.conv.dA[i], cache.conv.out[i-1], cache.conv.dK[i]);
    if(i != 0)  convolveATv(cache.conv.k[i], cache.conv.dA[i], cache.conv.dOut[i-1]); 
  }
}

void Net::apply_gradient(Cache& cache, size_t t) {
  for(size_t i = 0; i < cache.conv.count; i++) {
    conv_mtx[i].lock();
    adam(cache.conv.dK[i], model.conv_layers[i].emaK, model.conv_layers[i].maK, decay_rate1, decay_rate2, t);
    adam(cache.conv.dB[i], model.conv_layers[i].emaB, model.conv_layers[i].maB, decay_rate1, decay_rate2, t);

    L2(model.conv_layers[i].k, cache.conv.dK[i], regularization, learning_rate);
    L2(model.conv_layers[i].b, cache.conv.dB[i], regularization, learning_rate);
    conv_mtx[i].unlock();
  }

  for(size_t i = 0; i < cache.dense.count; i++) {
    dense_mtx[i].lock();
    adam(cache.dense.dW[i], model.dense_layers[i].emaW, model.dense_layers[i].maW, decay_rate1, decay_rate2, t);
    adam(cache.dense.dB[i], model.dense_layers[i].emaB, model.dense_layers[i].maB, decay_rate1, decay_rate2, t); 


    L2(model.dense_layers[i].w, cache.dense.dW[i], regularization, learning_rate);
    L2(model.dense_layers[i].b, cache.dense.dB[i], regularization, learning_rate);
    dense_mtx[i].unlock();
  }
}

void zeroGradients(Cache& cache) {
  for(size_t i = 0; i < cache.conv.count; i++) {
    zero(cache.conv.dK[i]);
    zero(cache.conv.dB[i]);
  }

  for(size_t i = 0; i < cache.dense.count; i++) {
    zero(cache.dense.dW[i]);
    zero(cache.dense.dB[i]);
  }
}

void Net::prepare_cache(const Matrix& X, int y, Cache& cache) {
  copyToTensorOfSameSize(X, cache.conv.out[-1]);
  cache.y = y;

  for(size_t i = 0; i < cache.conv.count; i++) {
    conv_mtx[i].lock();
    copyToTensorOfSameSize(model.conv_layers[i].k, cache.conv.k[i]);
    copyToTensorOfSameSize(model.conv_layers[i].b, cache.conv.b[i]);
    conv_mtx[i].unlock();
  }

  for(size_t i = 0; i < cache.dense.count; i++) {
    dense_mtx[i].lock();
    copyToTensorOfSameSize(model.dense_layers[i].w, cache.dense.w[i]);
    copyToTensorOfSameSize(model.dense_layers[i].b, cache.dense.b[i]);
    dense_mtx[i].unlock();
  }
}

void Net::train(Cache& cache, MnistReader& reader, int epoch, int t_index) {
  float entrsum = 0;
  reader.loop_to_beg();
  zeroGradients(cache);

  while(reader.read_next(AUGMENT)) {
    prepare_cache(reader.last_read, reader.last_lable, cache);
    forward_prop(cache);
    back_prop(cache);

    if((reader.index+1) % mini_batch == 0) {
      apply_gradient(cache, reader.number_of_entries*NTHREADS*epoch + reader.index*NTHREADS); 
      zeroGradients(cache);
      //drawConv(cache);
    }

    entrsum += crossEntropy(cache.dense.a[cache.dense.count-1].v, cache.y);

    if((reader.index+1) % PERIOD != 0) continue;
    if((reader.index+1) % (PERIOD*NTHREADS) == PERIOD*t_index) {
      std::cout << " executing sample " << reader.index+1 << " on thread " << t_index << " entropy average " << entrsum / PERIOD << "\n";
    }
    entrsum = 0;
  }
}

void Net::test_no_shinanegens(MnistReader& reader, Cache& cache) {
  size_t total_count = 0;
  for(size_t i = 0; i < reader.number_of_entries; i++) {
    prepare_cache(reader.images[i], reader.labels[i], cache);
    forward_prop(cache);
    Tensor<1>& preds = cache.dense.a[cache.dense.count -1];
    size_t max = std::distance(preds.v, std::max_element(preds.v, preds.v + preds.size));
    if(max == reader.labels.v[i]) total_count ++;
  }
  std::cout << "accuracy " << total_count / (float) reader.number_of_entries << "\n";
}

template<typename Function, typename CallBack>
void run_in_parallel(std::thread threads[], size_t n_threads, Function&& func, CallBack&& callback) {
  for (size_t t = 0; t < n_threads - 1; t++)
    threads[t] = std::thread(std::forward<Function>(func), t);

  std::forward<Function>(func)(n_threads-1);
  std::forward<CallBack>(callback)(n_threads-1);

  for (size_t t = 0; t < n_threads- 1; t++) {
    threads[t].join();
    std::forward<CallBack>(callback)(t);
  }
}

void Net::train_epochs(MnistReader& training_reader, int epochs, MnistReader& test_reader) {
  size_t control_size = training_reader.number_of_entries / 6;
  size_t sample_size = training_reader.number_of_entries - control_size;
  MnistReader sample_data = MnistReader(training_reader, 0, std::min<size_t>(10000, training_reader.number_of_entries));
  MnistReader control_data = MnistReader(training_reader, sample_size, training_reader.number_of_entries);

  std::thread threads[NTHREADS-1];

  //every thread has its own reader
  MnistReader* training_readers[NTHREADS];
  for(int t = 0; t < NTHREADS; t++) {
    training_readers[t] = new MnistReader(training_reader, t*sample_size/NTHREADS, (t+1)*sample_size/NTHREADS);
  }

  for(int e = 0; e < epochs; e++) {
    std::cout << "epoch " << e << "\n"; 
    run_in_parallel(threads, NTHREADS, [this, e, &training_readers](int t) {
      training_readers[t]->shuffle();
      this->train(threadscache[t], *training_readers[t], e, t);
    }, [](int t){ (void)t; });

    test_no_shinanegens(sample_data, threadscache[0]);
    test(sample_data, const_cast<char*>("smaple data accuracy: "));
    //test_no_shinanegens(control_data, threadscache[0]);
    test(control_data, const_cast<char*>("control data accuracy: "));
    //test_no_shinanegens(test_reader, threadscache[0]);
    test(test_reader, const_cast<char*>("test data accuracy: "));
  }

  std::cout << "training finished" << "\n"; 
}

void test(Cache& cache, MnistReader& reader) {
  size_t out_shape = cache.dense.a[cache.dense.count-1].size;
  TensorT<int, 1> count(out_shape);
  int total_correct = 0;

  for(size_t i = 0; i < reader.number_of_entries; i++) {
    copyToTensorOfSameSize(reader.images[i], cache.conv.out[-1]);
    cache.y = reader.labels[i];
    Vector preds = forward_prop(cache);
    int max = std::distance(preds.v, std::max_element(preds.v, preds.v + preds.size));

    count.v[reader.labels[i]]++;
    cache.results[reader.labels[i]][max]++; 
    if (max == (int) reader.labels[i]) total_correct++;
  }

  for(size_t x = 0; x < out_shape; x++) {
    for(size_t y = 0; y < out_shape; y++) {
      cache.results[x][y] = cache.results[x][y] / (count[x]+0.00001);
    }
  }

  cache.total_correct = total_correct;
}

float Net::test(MnistReader& reader, char* message) {
  std::thread threads[NTHREADS-1];

  size_t threads_lot = reader.number_of_entries / (float) NTHREADS;
  MnistReader* threads_readers[NTHREADS];
  for(int t = 0; t < NTHREADS; t++) 
    threads_readers[t] = new MnistReader(reader, t*threads_lot, (t+1)*threads_lot);
  
  size_t total_correct = 0;
  Tensor<2> total_results(threadscache[0].results.shape);

  run_in_parallel(threads, NTHREADS, [&threads_readers, this, &reader](int t) {
      prepare_cache(reader.images[0], reader.labels[0], threadscache[t]);
    ::test(threadscache[t], *threads_readers[t]);
  }, [this, &total_correct, &total_results](int t) { 
    total_correct += threadscache[t].total_correct; 
    addTens(threadscache[t].results, total_results);
  });

  std::cout << message << total_correct / (float) reader.number_of_entries << "\n";
  total_results * (1.0f/NTHREADS);
  drawMat(total_results);

  return total_correct / (float) reader.number_of_entries;
}


