#include <cmath>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <iterator>
#include <math.h>
#include <string>
#include <iostream>
#include <cstdlib>
#include <algorithm>
#include <bits/stdc++.h>
#include "net.hpp"
#include "math.hpp"
#include "../mnist_reader.hpp"

// activations --------------------------------------

void relu(float v[], int length) {
  for(int i = 0; i < length; i++) {
    v[i]=std::max<float>(0.0, v[i]);
  }
}

void relu_backward(float dA[], float A[], int length, float dZ[]) {
  for(int i = 0; i < length; i++) {
    dZ[i] = (A[i] > 0) * dA[i];
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

float crossEntropy(float *v, float *y, int length) {
  float sum = 0;
  for(int i = 0; i < length; i++) {
    sum -= y[i]*log(v[i]);
  }
  return sum;
}

// Net ----------------------------------------------

void Net::print_layer(int i, int thread_i) {
    DenseCache* caches = threadscache[thread_i];
    std::cout << "\nweights\n";
    printMat(layers[i].w);

    std::cout << "\nbiases\n";
    printMat(layers[i].b);

    std::cout << "\nact\n";
    printMat(caches[i].a);

    std::cout << "\ndWeights\n";
    printMat(caches[i].dW);

    std::cout << "\ndBiases\n";
    printMat(caches[i].dB);
}

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
  memset(dense.b.v, 0, dense.b.ht*sizeof(float));

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

Matrix& Net::forward_prop(Matrix& X, int thread_i) {
  DenseCache* caches = threadscache[thread_i]; // activations
  caches[0].a_prev = &X;
  
  for(int i = 0; i < layers_count; i++) {
    mulMat(layers[i].w, *(caches[i].a_prev), caches[i].a);
    addMat(layers[i].b, caches[i].a, caches[i].a);
    layers[i].activation(caches[i].a.v, caches[i].a.ht);
  }

  return caches[layers_count - 1].a;
}

void Net::train_epochs(MnistReader& reader, int epochs) {
  float Yv[layers[layers_count-1].out_shape];
  Matrix Y = {layers[layers_count-1].out_shape, 1, Yv};
  double entr_sum = 0;
  int t = 60;
  
  for(int e = 0; e < epochs; e++) {
    reader.loop_to_beg();

    for(int i = 0; i < reader.number_of_entries; i++) {
      reader.read_next();
      reader.last_read.ht = reader.last_read.ht*reader.last_read.wt;
      reader.last_read.wt = 1;

      memset(Y.v, 0, layers[layers_count-1].out_shape*sizeof(float));
      Y.v[reader.last_lable] = 1;

      train(reader.last_read, Y, t, 0);
      entr_sum -= crossEntropy(threadscache[0][layers_count-1].a.v, Y.v, Y.ht);

      if(i % 1000 == 0) {
        t++;
        float learning_rate = initial_learning_rate * sqrt(1 - pow(beta1, t)) / (1- pow(beta2, t));
        std::cout << "on epoch " << e << " entropy average: " << entr_sum / 1000.0 << " with learning rate: " << learning_rate << "\n";
        entr_sum =0;
      }
    }
  }
}

void Net::train(Matrix& X, Matrix& Y, int t, int thread_i) {
  forward_prop(X, thread_i);
  backward_prop(Y, thread_i);
  
  float learning_rate = initial_learning_rate * sqrt(1 - pow(beta1, t)) / (1- pow(beta2, t));
  learning_rate = std::min<float>(0.5, learning_rate);
  for(int i = 0; i < layers_count; i++) {
    addMat(layers[i].w, threadscache[thread_i][i].dW * -learning_rate, layers[i].w);
    addMat(layers[i].b, threadscache[thread_i][i].dB * -learning_rate, layers[i].b);
  }

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
}

void Net::test(MnistReader& reader, int thread_i) {
  int out_shape = threadscache[thread_i][layers_count -1].out_shape;
  float guessedv[out_shape*out_shape];
  Matrix guessed = {out_shape, out_shape, guessedv};
  memset(guessedv, 0, guessed.ht*guessed.wt*sizeof(int));

  float count[out_shape*out_shape];
  memset(count, 0, out_shape*sizeof(int));

  int total_correct;

  for(int i = 0; i < reader.number_of_entries; i++) {
    reader.read_next();
    reader.last_read.ht = reader.last_read.ht*reader.last_read.wt;
    reader.last_read.wt = 1;

    Matrix preds = forward_prop(reader.last_read, 0);
    auto max = std::distance(preds.v, std::max_element(preds.v, preds.v + preds.ht));

    count[reader.last_lable]++;
    guessed[reader.last_lable][max]++; 
    if (max == reader.last_lable) {
      total_correct++;
    }
  }

  for(int x = 0; x < out_shape; x++) {
    for(int y = 0; y < out_shape; y++) {
      guessed[x][y] = guessed[x][y] / count[x];
    }
  }
  
  std::cout << "guess count\n";
  printMat(guessed);
  std::cout << "drew\n";
  drawMat(guessed);

  std::cout << "\ntotal accuracy: " << static_cast<float>(total_correct) / static_cast<float>(reader.number_of_entries) << "\n";
}

