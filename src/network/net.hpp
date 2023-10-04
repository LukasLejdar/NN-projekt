#ifndef NET_H
#define NET_H
#include "math.hpp"

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

    Dense* layers;
    DenseCache* threadscache[NTHREADS]; // threadmem[thread_index][layer_index] = cache of layer

    Net(Dense layers[], int length);

    void forward_prop(Matrix& X, int thread_i);
    void backward_prop(Matrix& Y, int thread_i);
    void train(Matrix& X, int y, int thread_i);
};

void initialize_layer(Dense &dense);
void initialize_cache(DenseCache &cache);
void relu(float  v[], int length);
void sigmoid(float v[], int length);
void softmax(float v[], int length);



#endif
