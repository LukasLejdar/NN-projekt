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
  Matrix Y; 
  Matrix *a; //activations a[-1] is a duplicate of trained sample 
  Matrix *w, *b;
  Matrix *dB,*dW; 

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
    float learning_rate = 0.005;
    float decay_rate = 1;

    Net(Dense layers[], size_t length);

    ~Net() { delete [] layers; }

    static void func();
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

    void train(Cache* cache, MnistReader* reader, int t_index, int epoch);
    void apply_gradient(Cache& cache, int epoch, int sample);
};

void back_prop(Cache& cache);
Matrix& forward_prop(Cache& cache);
void relu(float  v[], int length);
void sigmoid(float v[], int length);
void softmax(float v[], int length);
float crossEntropy(float v[], float y[], int length);

#endif
