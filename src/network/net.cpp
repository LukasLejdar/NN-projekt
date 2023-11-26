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
  for(size_t i = 0; i < model.dense_count; i++) model.dense_layers[i].randomize();
  for(size_t i = 0; i < model.conv_count; i++) model.conv_layers[i].randomize();
}

Vector& forward_prop(Cache& cache) {
  assert(cache.conv.out[cache.conv.count-1].v == cache.dense.a[-1].v);

  for(size_t i = 0; i < cache.conv.count; i++) {
    correlateAv(cache.conv.k[i], cache.conv.out[i-1], cache.conv.a[i]);
    addTens(cache.conv.b[i], cache.conv.a[i]);
    maxPooling(cache.conv.a[i], cache.conv.pooling[i], cache.conv.out[i], cache.conv.loc[i]);
    relu(cache.conv.out[i].v, cache.conv.out[i].size);
  }
  
  for(size_t i = 0; i < cache.dense.count; i++) {
    matMulAv(cache.dense.w[i], cache.dense.a[i-1], cache.dense.a[i]);
    addTens(cache.dense.b[i], cache.dense.a[i]);
    if(i != cache.dense.count -1) relu(cache.dense.a[i].v, cache.dense.a[i].size);
  }

  softmax(cache.dense.a[cache.dense.count-1].v, cache.dense.a[cache.dense.count-1].size);
  return cache.dense.a[cache.dense.count -1];
}

void back_prop(Cache& cache) {
  assert(cache.dense.dA[-1].v == cache.conv.dOut[cache.conv.count-1].v );

  int i = cache.dense.count-1;
  cache.dense.dA[i].copyTensorOfSameSize(cache.dense.a[i]);
  cache.dense.dA[i].v[cache.y] -= 1;

  for(;i >= 0; i--) {
    addTens(cache.dense.dA[i], cache.dense.dB[i]);
    matMulvvT<false>(cache.dense.dA[i], cache.dense.a[i-1], cache.dense.dW[i]);
    matMulATv(cache.dense.w[i], cache.dense.dA[i], cache.dense.dA[i-1]);
    relu_backward(cache.dense.dA[i-1].v, cache.dense.a[i-1].v, cache.dense.a[i-1].size);
  }

  maxPooling_backward(cache.conv.dOut[cache.conv.count-1], cache.conv.dA[cache.conv.count-1], cache.conv.loc[cache.conv.count-1]);
  for(i = cache.conv.count-1; i >= 0; i--) {
    addTens(cache.conv.dA[i], cache.conv.dB[i]);
    correlatevvT<false>(cache.conv.dA[i], cache.conv.out[i-1], cache.conv.dK[i]);
    if(i != 0) convolveATv(cache.conv.k[i], cache.conv.dA[i], cache.conv.dOut[i-1]);
    if(i != 0) relu_backward(cache.conv.dOut[i-1].v, cache.conv.out[i-1].v, cache.conv.out[i-1].size);
    if(i != 0) maxPooling_backward(cache.conv.dOut[i-1], cache.conv.dA[i-1], cache.conv.loc[i-1]);
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

  for(size_t i = 0; i < cache.dense.count; i++) {
    dense_mtx[i].lock();
    rmsProp(cache.dense.dW[i], model.dense_layers[i].emaW, learning_rate, decay_rate1, t);
    rmsProp(cache.dense.dB[i], model.dense_layers[i].emaB, learning_rate, decay_rate1, t);

    addTens(cache.dense.dW[i], model.dense_layers[i].w);
    addTens(cache.dense.dB[i], model.dense_layers[i].b);
    dense_mtx[i].unlock();
  }
}

void zeroDerivatives(Cache& cache) {
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
  cache.conv.out[-1].copyTensorOfSameSize(X);
  cache.y = y;

  for(size_t i = 0; i < cache.conv.count; i++) {
    conv_mtx[i].lock();
    cache.conv.k[i].copyTensorOfSameSize(model.conv_layers[i].k);
    cache.conv.b[i].copyTensorOfSameSize(model.conv_layers[i].b);
    conv_mtx[i].unlock();
  }

  for(size_t i = 0; i < cache.dense.count; i++) {
    dense_mtx[i].lock();
    cache.dense.w[i].copyTensorOfSameSize(model.dense_layers[i].w);
    cache.dense.b[i].copyTensorOfSameSize(model.dense_layers[i].b);
    dense_mtx[i].unlock();
  }
}

void Net::train(Cache& cache, MnistReader& reader, int epoch, int t_index) {
  int period = 1000;
  float entrsum = 0;
  reader.loop_to_beg();
  zeroDerivatives(cache);

  while(reader.read_next()) {
    prepare_cache(reader.last_read, reader.last_lable, cache);
    forward_prop(cache);
    back_prop(cache);

    if(reader.index % mini_batch == 0) {
      apply_gradient(cache, reader.number_of_entries*NTHREADS*epoch + reader.index*NTHREADS); 
      zeroDerivatives(cache);
    }

    entrsum += crossEntropy(cache.dense.a[cache.dense.count-1].v, cache.y);

    if(reader.index % period != 0) continue;
    if(reader.index % (period*NTHREADS) == period*t_index)
      std::cout << " executing sample " << reader.index << " on thread " << t_index << " entropy average " << entrsum / period << "\n";
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
  std::thread threads[NTHREADS-1];

  MnistReader sample_data = MnistReader(training_reader, 0, std::min(1000, training_reader.number_of_entries));

  size_t control_size = training_reader.number_of_entries / 6;
  size_t sample_size = training_reader.number_of_entries - control_size;
  size_t test_size = test_reader.number_of_entries;

  //every thread has its own reader
  MnistReader* training_readers[NTHREADS];
  MnistReader* control_readers[NTHREADS];
  MnistReader* test_readers[NTHREADS];

  for(int t = 0; t < NTHREADS; t++) {
    training_readers[t] = new MnistReader(training_reader, t*sample_size/NTHREADS, (t+1)*sample_size/NTHREADS);
    control_readers[t] = new MnistReader(training_reader, t*control_size/NTHREADS, (t+1)*control_size/NTHREADS);
    test_readers[t] = new MnistReader(test_reader, t*test_size/NTHREADS, (t+1)*test_size/NTHREADS);
  }

  for(int e = 0; e < epochs; e++) {
    std::cout << "epoch " << e << "\n"; 
    run_in_parallel(threads, NTHREADS, [this, e, &training_readers](int t) {
      this->train(threadscache[t], *training_readers[t], e, t);
    }, [](int t){ (void)t; });

    if(e % 5 != 0) continue;
    test(sample_data);

    float total_accuracy = 0;
    run_in_parallel(threads, NTHREADS, [this, &control_readers](int t) {
      this->test(threadscache[t], *control_readers[t]);
    }, [this, &total_accuracy](int t) { 
      total_accuracy += threadscache[t].accuracy; 
    });

    std::cout << "control accuracy " << total_accuracy / NTHREADS << "\n";

    if(total_accuracy / NTHREADS > 0.88) {
      total_accuracy = 0;
      run_in_parallel(threads, NTHREADS, [this, &control_readers](int t) {
        this->test(threadscache[t], *control_readers[t]);
      }, [this, &total_accuracy](int t) { 
        total_accuracy += threadscache[t].accuracy; 
      });

      std::cout << "test accuracy " << total_accuracy / NTHREADS << "\n";
    }
  }

  for(auto tr_reader : training_readers) delete tr_reader;
  for(auto cr_reader : control_readers) delete cr_reader;
  for(auto test_reader : test_readers) delete test_reader;
  std::cout << "training finished" << "\n"; 
}

float Net::test(MnistReader& reader) {
  test(threadscache[0], reader);
  std::cout << "test accuracy: " << threadscache[0].accuracy << "\n";
  return threadscache[0].accuracy;
}

void Net::test(Cache& cache, MnistReader& reader) {
  reader.loop_to_beg();

  size_t out_shape = cache.dense.a[cache.dense.count-1].size;
  Matrix results = {out_shape, out_shape};
  TensorT<int, 2> count = {out_shape, 1};
  int total_correct = 0;

  while (reader.read_next()) {
    prepare_cache(reader.last_read, reader.last_lable, cache);
    Vector preds = forward_prop(cache);
    int max = std::distance(preds.v, std::max_element(preds.v, preds.v + preds.size));

    count.v[reader.last_lable]++;
    results[reader.last_lable][max]++; 
    if (max == reader.last_lable) total_correct++;
  }

  for(size_t x = 0; x < out_shape; x++) {
    for(size_t y = 0; y < out_shape; y++) {
      results[x][y] = results[x][y] / count[x][0];
    }
  }

  float percentage = static_cast<float>(total_correct) / reader.number_of_entries;
  cache.accuracy = percentage;
  
  //std::cout << "\ntotal accuracy: " << percentage << "\n";
  //std::cout << "guess count\n";
  //printMat(results);
  //drawMat(results, 1);
}


