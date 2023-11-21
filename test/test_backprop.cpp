#include <algorithm>
#include <assert.h>
#include <cstdio>
#include <iostream>
#include "../src/network/math.hpp"
#include "../src/network/net.hpp"
#include "test.hpp"

void prepare_cache(Cache& cache, Matrix* a_correct, Matrix* dW_correct, Matrix* dB_correct) { 
  const size_t LENGTH = 3;
  Dense layers[LENGTH] = {
    {5,4},
    {4,3},
    {3,2}
  };

  initialize_cache(cache, layers, LENGTH);

  float x[] = {0.23, 0.36, 0.08, 0.12, 0.81};
  cache.a[-1].setV(x);
  cache.y = 1;

  float w0[] = {
    -0.10, 0.32, -0.30, -0.57, 0.27,
    -0.56, -0.07, -0.37, -0.75, 0.25,
    0.63, 0.23, -0.20, 0.33, -0.00,
    -0.41, -0.33, -0.12, -0.48, -0.28,
  };
  float b0[] = {
    0.22,
    -0.39,
    0.00,
    -0.02,
  };
  float a0_correct[] = {
    0.4385,
    0.0000,
    0.2513,
    0.0000
  };
  float dB0_correct[] = {
    -0.317297, 
    0.00, 
    -0.159451, 
    0.00
  };
  float dW0_correct[] = {
    -0.0729783, -0.114227, -0.0253838, -0.0380756, -0.257011, 
    0, 0, 0, 0, 0, 
    -0.0366737, -0.0574024, -0.0127561, -0.0191341, -0.129155, 
    0, 0, 0, 0, 0
  };
  cache.w[0] = Matrix(4, 5, w0);
  cache.b[0] = Matrix(4, 1, b0);
  a_correct[0] = Matrix(4, 1, a0_correct);
  dB_correct[0] = Matrix(4, 1, dB0_correct);
  dW_correct[0] = Matrix(4, 5, dW0_correct);

  float w1[] = {
    0.34, -0.28, 0.19, -0.67,
    -0.03, -0.19, -0.55, 0.07,
    0.69, 0.57, -0.95, 0.24,
  };
  float b1[] = {
    0.06,
    -0.90,
    0.00,
  };
  float a1_correct[] = {
    0.256837,
    0.00,
    0.063830,
  };
  float dB1_correct[] = {
    -0.906085,
    0.0,
    -0.013374
  };
  float dW1_correct[] = {
    -0.397318, 0, -0.227699, 0, 
    0, 0, 0, 0, 
    -0.0058645, 0, -0.00336089, 0
  };
  cache.w[1] = Matrix(3, 4, w1);
  cache.b[1] = Matrix(3, 1, b1);
  a_correct[1] = Matrix(3, 1, a1_correct);
  dB_correct[1] = Matrix(3, 1, dB1_correct);
  dW_correct[1] = Matrix(3, 4, dW1_correct);
  
  float w2[] = {
    -1.14, -0.74, 0.24,
    1.57, 0.00, 0.28,
  };
  float b2[] = {
    0.00,
    -0.01,
  };
  float a2_correct[] = {
    0.3343488, 
    0.6656511,
  };
  float dB2_correct[] = {
    0.3343488, 
    -0.3343488, 
  };
  float dW2_correct[] = {
    0.0858731, 0, 0.0213415, 
    -0.0858731, 0, -0.0213415
  };
  cache.w[2] = Matrix(2, 3, w2);
  cache.b[2] = Matrix(2, 1, b2);
  a_correct[2] = Matrix(2, 1, a2_correct);
  dB_correct[2] = Matrix(2, 1, dB2_correct);
  dW_correct[2] = Matrix(2, 3, dW2_correct);
}

void test_back_prop() {
  float epsilon = 0.0001;
  Cache cache;

  Matrix *a_correct = new Matrix[3];
  Matrix *dW_correct = new Matrix[3];
  Matrix *dB_correct = new Matrix[3];

  prepare_cache(cache, a_correct, dW_correct, dB_correct);

  Matrix *dW_correcti = new Matrix[3];
  Matrix *dB_correcti = new Matrix[3];

  // minibatch of 10
  for(size_t i = 0; i < 2; i++) {
    zeroDerivatives(cache);
    for(size_t i = 0; i < 5; i++) {

      for(size_t l = 0; l < cache.layers_count; l++) {
        dB_correcti[l] = dB_correct[l];
        dW_correcti[l] = dW_correct[l];
        dB_correcti[l] * (i+1);
        dW_correcti[l] * (i+1);
      }

      std::cout << "\nprocessing sample " << i << "\n";
      forward_prop(cache);
      testMatOperation(new Matrix[]{cache.a[0], a_correct[0]}, "sample test activations 0", epsilon);
      testMatOperation(new Matrix[]{cache.a[1], a_correct[1]}, "test activations 1", epsilon);
      testMatOperation(new Matrix[]{cache.a[2], a_correct[2]}, "test activations 2", epsilon);

      back_prop(cache);
      testMatOperation(new Matrix[]{cache.dB[2], dB_correcti[2]}, "test dB 2", epsilon);
      testMatOperation(new Matrix[]{cache.dW[2], dW_correcti[2]}, "test dW 2", epsilon);
      testMatOperation(new Matrix[]{cache.dB[1], dB_correcti[1]}, "test dB 1", epsilon);
      testMatOperation(new Matrix[]{cache.dW[1], dW_correcti[1]}, "test dW 1", epsilon);
      testMatOperation(new Matrix[]{cache.dB[0], dB_correcti[0]}, "test dB 0", epsilon);
      testMatOperation(new Matrix[]{cache.dW[0], dW_correcti[0]}, "test dW 0", epsilon);


    }
  }
}

int main(void) {
  test_back_prop();
  return 0;
}

