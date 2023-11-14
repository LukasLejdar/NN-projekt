#ifndef NET_H
#define NET_H

#include <mutex>
#include <array>
#include "math.hpp"
#include "../mnist_reader.hpp"

#define RELU 0
#define SIGMOID 1
#define SOFTMAX 2
#define NTHREADS 4


struct Dense {
  size_t in_shape, out_shape;
  Matrix w;
  Matrix b;

  Dense(): in_shape(0), out_shape(0), w({}), b({}) {}
  Dense(const Dense& other);
  Dense(size_t in_shape, size_t out_shape);
  void swap(Dense& other);
  Dense& operator=(const Dense& other);
};

struct Cache {
  Matrix* a; //activations a[-1] is duplicate of trained sample 
  Matrix* w;
  Matrix* b;
  Matrix* dB; 
  Matrix* dW;
  Matrix Y;

  ~Cache() {
    delete [] (a-1);
    delete [] dB; 
    delete [] dW;
    delete [] w;
    delete [] b;
  };
};

class Net {
  public:
    size_t layers_count;
    float learning_rate = 0.001;

    Net(Dense layers[], size_t length);

    ~Net() { delete [] layers; }

    static void func();
    Matrix& forward_prop(Cache& cache);
    void train_epochs(MnistReader& reader, int epochs);
    void test(MnistReader& reader);
    void print_layer(size_t i, Cache& cache);
    void print_layer(size_t i, size_t t);
    void initialize_cache(Cache& cache);
    void prepare_cache(Matrix& X, int y, Cache& cache);

  private:
    std::mutex* mtx;
    Dense* layers;
    Cache threadscache[NTHREADS];

    void train(Cache* cache, MnistReader* reader, int t_index);
    void apply_gradient(Cache& cache, int t_index);
    void back_prop(Cache& cache);
};

void relu(float  v[], int length);
void sigmoid(float v[], int length);
void softmax(float v[], int length);
float crossEntropy(float v[], float y[], int length);

#endif
