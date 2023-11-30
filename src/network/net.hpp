#ifndef NET_H
#define NET_H

#include <mutex>
#include <array>
#include "math.hpp"
#include "layer.hpp"
#include "../mnist_reader.hpp"

#define NTHREADS 8
#define PERIOD 1000
#define AUGMENT false 

class Net {
  public:
    size_t mini_batch = 48;
    float learning_rate = 0.001;
    float decay_rate1 = 0.9;
    float decay_rate2 = 0.99;
    float regularization = 0;

    Net(Model& model);

    void prepare_cache(const Matrix& X, int y, Cache& cache);
    void train_epochs(MnistReader& reader, int epochs, MnistReader& test_reader);
    void train(Cache& cache, MnistReader& reader, int epoch, int t_index);
    float test(MnistReader& reader, char* message);
    void test_no_shinanegens(MnistReader& reader, Cache& cache);

  private:
    std::mutex* conv_k_mtx, *conv_b_mtx;
    std::mutex* dense_w_mtx, *dense_b_mtx;

    Model& model;
    Cache threadscache[NTHREADS];

    void apply_gradient(Cache& cache, size_t t);
};

Vector& forward_prop(Cache& cache);
void back_prop(Cache& cache);
void zeroGradients(Cache& cache);

#endif
