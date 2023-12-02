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
  conv_b_mtx = new std::mutex[model.conv_count];
  conv_k_mtx = new std::mutex[model.conv_count];
  dense_w_mtx = new std::mutex[model.dense_count];
  dense_b_mtx = new std::mutex[model.dense_count];

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
    conv_k_mtx[i].lock();
    rmsProp(cache.conv.dK[i], model.conv_layers[i].emaK, decay_rate1, t);
    L2(model.conv_layers[i].k, cache.conv.dK[i], regularization, learning_rate);
    conv_k_mtx[i].unlock();
  }

  for(size_t i = 0; i < cache.dense.count; i++) {
    dense_w_mtx[i].lock();
    rmsProp(cache.dense.dW[i], model.dense_layers[i].emaW, decay_rate1, t);
    L2(model.dense_layers[i].w, cache.dense.dW[i], regularization, learning_rate);
    dense_w_mtx[i].unlock();
  }

  for(size_t i = 0; i < cache.conv.count; i++) {
    conv_b_mtx[i].lock();
    rmsProp(cache.conv.dB[i], model.conv_layers[i].emaB, decay_rate1, t);
    L2(model.conv_layers[i].b, cache.conv.dB[i], regularization, learning_rate);
    conv_b_mtx[i].unlock();
  }

  for(size_t i = 0; i < cache.dense.count; i++) {
    dense_b_mtx[i].lock();
    rmsProp(cache.dense.dB[i], model.dense_layers[i].emaB, decay_rate1, t); 
    L2(model.dense_layers[i].b, cache.dense.dB[i], regularization, learning_rate);
    dense_b_mtx[i].unlock();
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

void Net::copy_model_to_cache(Cache& cache) {

  for(size_t i = 0; i < cache.conv.count; i++) {
    conv_k_mtx[i].lock();
    copyToTensorOfSameSize(model.conv_layers[i].k, cache.conv.k[i]);
    conv_k_mtx[i].unlock();
  }

  for(size_t i = 0; i < cache.conv.count; i++) {
    conv_b_mtx[i].lock();
    copyToTensorOfSameSize(model.conv_layers[i].b, cache.conv.b[i]);
    conv_b_mtx[i].unlock();
  }

  for(size_t i = 0; i < cache.dense.count; i++) {
    dense_w_mtx[i].lock();
    copyToTensorOfSameSize(model.dense_layers[i].w, cache.dense.w[i]);
    dense_w_mtx[i].unlock();
  }

  for(size_t i = 0; i < cache.dense.count; i++) {
    dense_b_mtx[i].lock();
    copyToTensorOfSameSize(model.dense_layers[i].b, cache.dense.b[i]);
    dense_b_mtx[i].unlock();
  }
}

void Net::train(Cache& cache, MnistReader& reader, int epoch, int t_index) {
  float entrsum = 0;
  reader.loop_to_beg();
  zeroGradients(cache);

  while(reader.read_next(AUGMENT, cache.conv.out[-1], cache.y)) {
    forward_prop(cache);
    back_prop(cache);

    if((reader.index+1) % mini_batch == 0) {
      apply_gradient(cache, (reader.number_of_entries*epoch + reader.index)*NTHREADS); 
      zeroGradients(cache);
      copy_model_to_cache(cache);
    }

    entrsum += crossEntropy(cache.dense.a[cache.dense.count-1].v, cache.y);

    if((reader.index+1) % PERIOD != 0) continue;
    if((reader.index+1) % (PERIOD*NTHREADS) == PERIOD*t_index) {
      std::cout << " executing sample " << reader.index+1 << " on thread " << t_index << " entropy average " << entrsum / PERIOD << "\n";
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

void Net::train_epochs(MnistReader& training_reader, int epochs, MnistReader& test_reader, float threashold) {
  std::thread threads[NTHREADS-1];
  training_reader.shuffle();

  size_t control_size = training_reader.number_of_entries / 5;
  size_t sample_size = training_reader.number_of_entries - control_size;
  MnistReader sample_reader(training_reader, 0, sample_size);
  MnistReader control_reader(training_reader, sample_size, training_reader.number_of_entries);

  //every thread has its own reader
  MnistReader sample_readers[NTHREADS];
  for(int t = 0; t < NTHREADS; t++) {
    new (&sample_readers[t]) MnistReader(sample_reader, t * sample_size / NTHREADS, (t+1) * sample_size / NTHREADS); 
  }

  for(int e = 0; e < epochs; e++) {
    std::cout << "epoch " << e << "\n"; 
    sample_reader.shuffle();
    run_in_parallel(threads, NTHREADS, [this, e, &sample_readers](int t) {
      this->train(threadscache[t], sample_readers[t], e, t);
    }, [](int t){ (void)t; });

    if(e == 8) learning_rate /= 2;
    if(e < 10) continue;

    test(sample_readers[0], const_cast<char*>("smaple data accuracy: "));
    float accuracy = test(control_reader, const_cast<char*>("control data accuracy: "));
    if(accuracy >= threashold) return;
    test(test_reader, const_cast<char*>("test data accuracy: "));
  }

  forward_prop(threadscache[0]);
  back_prop(threadscache[0]);
  drawConv(threadscache[0]);

  std::cout << "training finished" << "\n\n";
}


// Tests ----------------------------------------------------------------------

int predict(Cache& cache) {
  Vector preds = forward_prop(cache);
  return std::distance(preds.v, std::max_element(preds.v, preds.v + preds.size));
}

float Net::test(MnistReader& reader, char* message) {
  std::thread threads[NTHREADS-1];

  size_t out_shape = model.dense_layers[model.dense_count-1].out_shape.size;
  Tensor<2> total_results(out_shape, out_shape);
  TensorT<int, 1> total_labels_count(out_shape);

  size_t threads_lot = reader.number_of_entries / (float) NTHREADS;
  MnistReader threads_readers[NTHREADS];
  for(int t = 0; t < NTHREADS; t++) {
    new(&threads_readers[t])  MnistReader(reader, t*threads_lot, (t+1)*threads_lot);
  }

  run_in_parallel(threads, NTHREADS, [&threads_readers, this, &reader](int t) {
      Cache& cache = threadscache[t];

      threads_readers[t].loop_to_beg();
      copy_model_to_cache(cache);
      zero(cache.labels_count);
      zero(cache.results);

      while(threads_readers[t].read_next(false, cache.conv.out[-1], cache.y)) {
        int pred = predict(cache);
        cache.labels_count.v[cache.y]++;
        cache.results[cache.y][pred]++; 
      }
  }, [this, &total_results, &total_labels_count](int t) { 
      addTens(threadscache[t].results, total_results);
      addTens(threadscache[t].labels_count, total_labels_count);
  });

  size_t total_correct = 0;
  for(size_t x = 0; x < out_shape; x++) {
    total_correct += total_results[x][x];
    for(size_t y = 0; y < out_shape; y++) {
      total_results[x][y] = total_results[x][y] * out_shape / (total_labels_count[x]+0.00001);
    }
  }

  std::cout << message << total_correct / (float) reader.number_of_entries << "\n";

  total_results * (1.0f/NTHREADS);
  drawMat(total_results);
  std::cout << "\n";
  return total_correct / (float) reader.number_of_entries;
}

void Net::make_preds(const Tensor<3>& images, std::string preds_path) {
    std::thread threads[NTHREADS-1];
    TensorT<int, 1> preds = TensorT<int, 1>(images.shape[0]);
    for(size_t t = 0; t < NTHREADS; t++) {
      copy_model_to_cache(threadscache[t]);
    }

    run_in_parallel(threads, NTHREADS, [this, &images, &preds](int t) {
      size_t end = (t == NTHREADS-1) ? images.shape[0] : (t+1)*images.shape[0] / NTHREADS;
      Tensor<3> threads_images = images.reference(t*images.shape[0] / NTHREADS, end);
      TensorT<int, 1> threads_preds = preds.reference(t*images.shape[0] / NTHREADS, end);

      for(size_t i = 0; i < threads_images.shape[0]; i++) {
        copyToTensorOfSameSize(threads_images[i], threadscache[t].conv.out[-1]);
        threads_preds.v[i] = predict(threadscache[t]);
      }
    }, [this](int t) { (void)t; });

    std::ofstream preds_file(preds_path, std::ios::trunc);

    if (!preds_file.is_open()) {
      std::cerr << "\033[48;2;255;0;0m Unable to open file " << preds_path << " \033[0m" << "\n";
      return;
    }

    for(size_t i = 0; i < preds.size; i++) {
      preds_file << preds.v[i] << "\n";
    }

    preds_file.close();
    std::cout << "Successfully wrote results to " << preds_path << std::endl;
}


