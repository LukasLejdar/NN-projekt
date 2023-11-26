#ifndef NET_H
#define NET_H

#include <mutex>
#include <array>
#include "math.hpp"
#include "layer.hpp"
#include "../mnist_reader.hpp"

#define NTHREADS 8

class Net {
  public:
    size_t mini_batch = 30;
    float learning_rate = 0.001;
    float decay_rate1 = 0.9;
    float decay_rate2 = 0.99;

    Net(Model& model);

    void prepare_cache(Matrix& X, int y, Cache& cache);
    void train_epochs(MnistReader& reader, int epochs, MnistReader& test_reader);
    void test(Cache& cache, MnistReader& reader);
    float test(MnistReader& reader);

  private:
    std::mutex* dense_mtx;
    std::mutex* conv_mtx;
    Model& model;
    Cache threadscache[NTHREADS];

    void train(Cache& cache, MnistReader& reader, int epoch, int t_index);
    void apply_gradient(Cache& cache, size_t t);
};

Vector& forward_prop(Cache& cache);
void back_prop(Cache& cache);
void zeroDerivatives(Cache& cache);

#endif
