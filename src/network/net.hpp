#include "math.hpp"
#include "../mnist_reader.hpp"


#ifndef NET_H
#define NET_H

#define RELU 0
#define SIGMOID 1
#define SOFTMAX 2
#define NTHREADS 10

#define DENSE(in, out, activ) {in, out, activ, nullptr, nullptr, {}, {}}

struct Dense {
  int in_shape; int out_shape;
  int acti;
  void (*activation) (float v[], int length);
  void (*back_activation) (float dA[], float A[], int length, float dZ[]); //save result to dZ
  Matrix w;
  Matrix b;
};

struct DenseCache {
  int in_shape, out_shape;
  Matrix* a_prev; //activations
  Matrix a; 
  Matrix dW;
  Matrix dB;
};

class Net {
  public:
    int layers_count;
    float initial_learning_rate = 0.001;
    float decay_rate = 0.5;
    float beta1 = 0.9;
    float beta2 = 0.999;

    Dense* layers;
    DenseCache* threadscache[NTHREADS]; // threadmem[thread_index][layer_index] = cache of layer

    Net(Dense layers[], int length);

    Matrix& forward_prop(Matrix& X, int thread_i);
    void backward_prop(Matrix& Y, int thread_i);
    void train(Matrix& X, Matrix& y, int epoch, int thread_i);
    void train_epochs(MnistReader& reader, int epochs);
    void test(MnistReader& reader, int thread_i);
    void print_layer(int i, int threadi);
};

void initialize_layer(Dense &dense);
void initialize_cache(DenseCache &cache);
void relu(float  v[], int length);
void sigmoid(float v[], int length);
void softmax(float v[], int length);
float crossEntropy(float v[], float y[], int length);




#endif
