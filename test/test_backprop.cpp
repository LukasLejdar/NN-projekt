#include <algorithm>
#include <assert.h>
#include <cstdio>
#include <iostream>
#include "../src/network/math.hpp"
#include "../src/network/net.hpp"
#include "test.hpp"

void prepare_cache(Cache& cache, 
    Matrix* a_correct, 
    Matrix* dW_correct,
    Matrix* dB_correct) { 

  cache.a = new Matrix[cache.layers_count+1] + 1;
  cache.dW = new Matrix[cache.layers_count];
  cache.dB = new Matrix[cache.layers_count];
  cache.w = new Matrix[cache.layers_count];
  cache.b = new Matrix[cache.layers_count];

  cache.a[0] = Matrix(4, 1);
  cache.dB[0] = Matrix(4, 1);
  cache.dW[0] = Matrix(4, 5);

  cache.a[1] = Matrix(3, 1);
  cache.dB[1] = Matrix(3, 1);
  cache.dW[1] = Matrix(3, 4);

  cache.a[2] = Matrix(2, 1);
  cache.dB[2] = Matrix(2, 1);
  cache.dW[2] = Matrix(2, 3);

  float x[] = {0.23, 0.36, 0.08, 0.12, 0.81};
  cache.a[-1] = Matrix(5,1,x);
  float y[] = {0, 1};
  cache.Y = Matrix(2,1,y);

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
  cache.layers_count = 3;

  Matrix *a_correct = new Matrix[cache.layers_count];
  Matrix *dW_correct = new Matrix[cache.layers_count];
  Matrix *dB_correct = new Matrix[cache.layers_count];

  prepare_cache(cache, a_correct, dW_correct, dB_correct);
  
  forward_prop(cache);
  testMatOperation(new Matrix[]{cache.a[0], a_correct[0]}, "test activations 0", epsilon);
  testMatOperation(new Matrix[]{cache.a[1], a_correct[1]}, "test activations 1", epsilon);
  testMatOperation(new Matrix[]{cache.a[2], a_correct[2]}, "test activations 2", epsilon);

  back_prop(cache);
  testMatOperation(new Matrix[]{cache.dB[2], dB_correct[2]}, "test dB 2", epsilon);
  testMatOperation(new Matrix[]{cache.dW[2], dW_correct[2]}, "test dW 2", epsilon);
  testMatOperation(new Matrix[]{cache.dB[1], dB_correct[1]}, "test dB 1", epsilon);
  testMatOperation(new Matrix[]{cache.dW[1], dW_correct[1]}, "test dW 1", epsilon);
  testMatOperation(new Matrix[]{cache.dB[0], dB_correct[0]}, "test dB 0", epsilon);
  testMatOperation(new Matrix[]{cache.dW[0], dW_correct[0]}, "test dW 0", epsilon);
}

int main(void) {
  test_back_prop();
  return 0;
}

