#include "math.hpp"
#include "../mnist_reader.hpp"

#ifndef NET_H
#define NET_H

#define RELU 0
#define SIGMOID 1
#define SOFTMAX 2
#define NTHREADS 10

#define DENSE(in, out) {in, out, {}, {}}

struct Dense {
  int in_shape; int out_shape;
  Matrix w;
  Matrix b;
  Matrix dW;
  Matrix dB;

};

struct ThreadsCache {
  Matrix a; 
};

class Net {
  public:
    int layers_count;
    float initial_learning_rate = 0.001;
    float decay_rate = 0.5;
    float beta1 = 0.9;
    float beta2 = 0.999;
    int batch_count = 5;

    Dense* layers;
    ThreadsCache* threadscache[NTHREADS];

    Net(Dense layers[]);

    Matrix& forward_prop(Matrix& X, ThreadsCache* cache);
    void backward_prop(Matrix& Y, ThreadsCache* cache);
    void train(Matrix& X, Matrix& y, ThreadsCache* cache);
    void train_epochs(MnistReader& reader, int epochs);
    void test(MnistReader& reader, ThreadsCache* cache);
    void print_layer(int i, ThreadsCache* cache);
};

void initialize_layer(Dense &dense);
void initialize_cache(ThreadsCache &cache);
void relu(float  v[], int length);
void sigmoid(float v[], int length);
void softmax(float v[], int length);
float crossEntropy(float v[], float y[], int length);

#endif
