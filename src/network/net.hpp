#ifndef NET_H
#define NET_H

#include <mutex>
#include <array>
#include "math.hpp"
#include "../mnist_reader.hpp"
#include "layer.hpp"

#define NTHREADS 6

struct Cache {
  size_t layers_count;
  size_t y; // corect label
  Matrix *a; //activations a[-1] is a duplicate of trained sample 
  Matrix *w, *b;
  Matrix *dB,*dW; 
  Matrix *dA;

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
    size_t mini_batch = 30;
    float learning_rate = 0.001;
    float decay_rate1 = 0.9;
    float decay_rate2 = 0.99;

    Net(Dense layers[], size_t length);

    ~Net() { delete [] layers; }

    static void func();
    void train_epochs(MnistReader& reader, int epochs, MnistReader& test_reader);
    float test(MnistReader& reader);
    void print_layer(size_t i, Cache& cache);
    void print_layer(size_t i, size_t t);
    void prepare_cache(Matrix& X, int y, Cache& cache);

  private:
    std::mutex* mtx;
    Dense* layers;
    Cache threadscache[NTHREADS];

    void train(Cache* cache, MnistReader* reader, int t_index, int epoch);
    void apply_gradient(Cache& cache, size_t t);
};

void back_prop(Cache& cache);
Matrix& forward_prop(Cache& cache);
void relu(float  v[], int length);
void sigmoid(float v[], int length);
void softmax(float v[], int length);
float crossEntropy(float v[], float y[], int length);
void initialize_cache(Cache& cache, Dense layers[], size_t layers_count);
void zeroDerivatives(Cache& cache);

#endif
