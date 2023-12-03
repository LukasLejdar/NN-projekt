#ifndef NET_H
#define NET_H

#include <mutex>
#include <array>
#include "math.hpp"
#include "layer.hpp"
#include "../mnist_reader.hpp"

#define NTHREADS 8
#define PERIOD 1000
#define AUGMENT true 
 
class Net {
  public:
    size_t mini_batch = 128;
    float learning_rate = 0.001;
    float decay_rate1 = 0.9;
    float decay_rate2 = 0.99;
    float regularization = 0.00007;

    Net(Model& model);

    void copy_model_to_cache(Cache& cache);
    void train_epochs(MnistReader& reader, int epochs, float threashold);
    void train(Cache& cache, MnistReader& reader, int epoch, int t_index);
    void make_preds(const Tensor<3>& images, std::string preds_path);
    float test(MnistReader& reader, char* message);

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
