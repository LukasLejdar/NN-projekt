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
  for(size_t t = 0; t < NTHREADS+1; t++) initialize_cache(threadscache[t], model);
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
    rmsProp(cache.conv.dK[i], model.conv_layers[i].emaK, learning_rate, decay_rate1, t);
    rmsProp(cache.conv.dB[i], model.conv_layers[i].emaB, learning_rate, decay_rate1, t);

    addTens(cache.conv.dK[i], model.conv_layers[i].k);
    addTens(cache.conv.dB[i], model.conv_layers[i].b);
    conv_mtx[i].unlock();
  }

  for(size_t i = 0; i < cache.dense.count-1; i++) {
    dense_mtx[i].lock();
    rmsProp(cache.dense.dW[i], model.dense_layers[i].emaW, learning_rate, decay_rate1, t);
    rmsProp(cache.dense.dB[i], model.dense_layers[i].emaB, learning_rate, decay_rate1, t);

    addTens(cache.dense.dW[i], model.dense_layers[i].w);
    addTens(cache.dense.dB[i], model.dense_layers[i].b);
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

void Net::prepare_cache(Matrix& X, int y, Cache& cache) {
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

void test_kGrad(Cache& cache) {
  std::cout << "\nbefore k\n";
  drawKernels(cache.conv.k[0]);
  const Tensor<4> true_dK(cache.conv.k[0].shape);
  const Tensor<4> k_orig(cache.conv.k[0].shape);
  copyToTensorOfSameSize(cache.conv.k[0], k_orig);

  for(size_t i = 0; i < cache.conv.k[0].size; i++) {
    copyToTensorOfSameSize(k_orig, cache.conv.k[0]);
    float e = 0.0001;
    cache.conv.k[0].v[i] += e;

    forward_prop(cache);
    float entropy1 = crossEntropy(cache.dense.a[cache.dense.count-1].v, cache.y);

    copyToTensorOfSameSize(k_orig, cache.conv.k[0]);
    cache.conv.k[0].v[i] -= e;
    forward_prop(cache);
    float entropy2 = crossEntropy(cache.dense.a[cache.dense.count-1].v, cache.y);

    true_dK.v[i] = (entropy1 - entropy2) /(2*e);
  }
  
  copyToTensorOfSameSize(k_orig, cache.conv.k[0]);

  std::cout << "\nafter k\n";
  drawKernels(cache.conv.k[0]);
  std::cout << "\ntest dK\n";
  drawKernels(cache.conv.dK[0]);
  std::cout << "\ntrue dK\n";
  drawKernels(true_dK);
}

void Net::train(Cache& cache, MnistReader& reader, int epoch, int t_index) {
  int period = 1000;
  float entrsum = 0;
  reader.loop_to_beg();
  zeroGradients(cache);

  while(reader.read_next()) {
    prepare_cache(reader.last_read, reader.last_lable, cache);
    forward_prop(cache);
    back_prop(cache);

    //if(reader.index % mini_batch == 0) {
    //  test_kGrad(cache);
    //}

    if(reader.index % mini_batch == 0) {
      apply_gradient(cache, reader.number_of_entries*NTHREADS*epoch + reader.index*NTHREADS); 
      zeroGradients(cache);
    }

    entrsum += crossEntropy(cache.dense.a[cache.dense.count-1].v, cache.y);

    if(reader.index % (period*NTHREADS) == (period*t_index + epoch)) {
      //drawConv(cache);
    }

    if(reader.index % period != 0) continue;

    if(reader.index % (period*NTHREADS) == period*t_index) {
      std::cout << " executing sample " << reader.index << " on thread " << t_index << " entropy average " << entrsum / period << "\n";
    }
    entrsum = 0;
  }
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
  MnistReader control_data = MnistReader(training_reader, sample_size, training_reader.number_of_entries);
  MnistReader sample_data = MnistReader(training_reader, 0, std::min(5000, training_reader.number_of_entries));

  std::thread threads[NTHREADS-1];

  //every thread has its own reader
  MnistReader* training_readers[NTHREADS];
  for(int t = 0; t < NTHREADS; t++) {
    training_readers[t] = new MnistReader(training_reader, t*sample_size/NTHREADS, (t+1)*sample_size/NTHREADS);
  }

  for(int e = 0; e < epochs; e++) {
    std::cout << "epoch " << e << "\n"; 
    run_in_parallel(threads, NTHREADS, [this, e, &training_readers](int t) {
      this->train(threadscache[t], *training_readers[t], e, t);
    }, [](int t){ (void)t; });

    if(e % 1 != 0) continue;
    std::cout << "\n";
    test(sample_data, const_cast<char*>("smaple data accuracy: "));
    test(control_data, const_cast<char*>("control data accuracy: "));
    test(test_reader, const_cast<char*>("test data accuracy: "));
  }

  std::cout << "training finished" << "\n"; 
}

void test(Cache& cache, MnistReader& reader) {
  reader.loop_to_beg();

  size_t out_shape = cache.dense.a[cache.dense.count-1].size;
  Matrix results(out_shape, out_shape);
  TensorT<int, 1> count(out_shape);
  int total_correct = 0;

  while (reader.read_next()) {
    copyToTensorOfSameSize(reader.last_read, cache.conv.out[-1]);
    cache.y = reader.last_lable;
    Vector preds = forward_prop(cache);
    int max = std::distance(preds.v, std::max_element(preds.v, preds.v + preds.size));

    count.v[reader.last_lable]++;
    results[reader.last_lable][max]++; 
    if (max == reader.last_lable) total_correct++;
  }

  for(size_t x = 0; x < out_shape; x++) {
    for(size_t y = 0; y < out_shape; y++) {
      results[x][y] = results[x][y] / count[x];
    }
  }

  float percentage = static_cast<float>(total_correct) / reader.number_of_entries;
  cache.accuracy = percentage;
  
  //std::cout << "\ntotal accuracy: " << percentage << "\n";
  //std::cout << "guess count\n";
  //printMat(results);
  //drawMat(results, 1);
}

float Net::test(MnistReader& reader, char* message) {
  std::thread threads[NTHREADS-1];

  size_t threads_lot = reader.number_of_entries / NTHREADS;
  MnistReader* threads_readers[NTHREADS];
  for(int t = 0; t < NTHREADS; t++) 
    threads_readers[t] = new MnistReader(reader, t*threads_lot/NTHREADS, (t+1)*threads_lot/NTHREADS);
  

  float total_accuracy = 0;
  run_in_parallel(threads, NTHREADS, [&threads_readers, this](int t) {
    ::test(threadscache[t], *threads_readers[t]);
  }, [this, &total_accuracy](int t) { 
    total_accuracy += threadscache[t].accuracy; 
  });

  std::cout << message << total_accuracy / NTHREADS << "\n";
  return threadscache[0].accuracy;
}


//void test_bGrad(Cache& cache) {
//  std::cout << "\nbefore b\n";
//  draw3D(cache.conv.b[0]);
//  const Tensor<3> true_dB(cache.conv.b[0].shape);
//  const Tensor<3> b_orig(cache.conv.b[0].shape);
//  copyToTensorOfSameSize(cache.conv.b[0], b_orig);
//
//  for(size_t i = 0; i < cache.conv.b[0].size; i++) {
//    copyToTensorOfSameSize(b_orig, cache.conv.b[0]);
//    float e = 0.0001;
//    cache.conv.b[0].v[i] += e;
//
//    forward_prop(cache);
//    float entropy1 = crossEntropy(cache.dense.a[cache.dense.count-1].v, cache.y);
//
//    copyToTensorOfSameSize(b_orig, cache.conv.b[0]);
//    cache.conv.b[0].v[i] -= e;
//    forward_prop(cache);
//    float entropy2 = crossEntropy(cache.dense.a[cache.dense.count-1].v, cache.y);
//
//    true_dB.v[i] = (entropy1 - entropy2) /(2*e);
//  }
//  
//  copyToTensorOfSameSize(b_orig, cache.conv.b[0]);
//
//  std::cout << "\nafter b\n";
//  draw3D(cache.conv.b[0]);
//  std::cout << "\ntest dB\n";
//  draw3D(cache.conv.dB[0]);
//  std::cout << "\ntrue dB\n";
//  draw3D(true_dB);
//}
//
//Vector& forward_propOut(Cache& cache) {
//  assert(cache.conv.out[cache.conv.count-1].v == cache.dense.a[-1].v);
//
//  for(size_t i = 0; i < cache.dense.count; i++) {
//    matMulAv(cache.dense.w[i], cache.dense.a[i-1], cache.dense.a[i]);
//    addTens(cache.dense.b[i], cache.dense.a[i]);
//    if(i != cache.dense.count -1) relu(cache.dense.a[i]);
//  }
//
//  softmax(cache.dense.a[cache.dense.count-1]);
//  return cache.dense.a[cache.dense.count -1];
//}
//
//void test_outGradient(Cache& cache) {
//  size_t k = cache.conv.count-1;
//
//  const Tensor<3> true_dOut(cache.conv.out[k].shape);
//  const Tensor<3> out_orig(cache.conv.out[k].shape);
//  copyToTensorOfSameSize(cache.conv.out[k], out_orig);
//
//  for(size_t i = k; i < cache.conv.out[0].size; i++) {
//    copyToTensorOfSameSize(out_orig, cache.conv.out[k]);
//    float e = 0.0001;
//    cache.conv.out[k].v[i] += e;
//
//    forward_propOut(cache);
//    float entropy1 = crossEntropy(cache.dense.a[cache.dense.count-1].v, cache.y);
//
//    copyToTensorOfSameSize(out_orig, cache.conv.out[k]);
//    cache.conv.out[k].v[i] -= e;
//    forward_propOut(cache);
//    float entropy2 = crossEntropy(cache.dense.a[cache.dense.count-1].v, cache.y);
//
//    true_dOut.v[i] = (entropy1 - entropy2) /(2*e);
//  }
//  
//  copyToTensorOfSameSize(out_orig, cache.conv.out[k]);
//
//  std::cout << "\nafter out\n";
//  draw3D(cache.conv.out[k]);
//  std::cout << "\ntest dOut\n";
//  draw3D(cache.conv.dOut[k]);
//  std::cout << "\ntrue dOut\n";
//  draw3D(true_dOut);
//}
