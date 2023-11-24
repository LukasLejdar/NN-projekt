#include <algorithm>
#include <algorithm>
#include <assert.h>
#include <cstddef>
#include <cstdio>
#include <iostream>
#include "test.hpp"
#include "../src/network/math.hpp"
#include "../src/network/net.hpp"
#include "../src/mnist_reader.hpp"

Matrix* getTransposeTest0() {
  float v[] = { 
    0, 2, 3, 
    0 ,0, 0,
    1, 2, 4};
  Matrix m(v,3,3);
  float correct[] = {
    0, 0, 1,
    2, 0, 2,
    3, 0, 4};
  Matrix *list = new Matrix[]{{3,3}, {correct, 3,3}};
  transpose<8>(m, list[0]);
  return list;
}

Matrix* getTransposeTest1() {
  float v[] = { 
    1, 
    2,
    3};
  Matrix m(v,3,1);
  float* correct = {new float[]{
    1, 2, 3}};
  Matrix *list = new Matrix[]{{1,3}, {correct,1,3}};
  transpose<8>(m, list[0]);
  return list;
}

Matrix* getTransposeTest2() {
  float v[] = { 
    1, 2, 3,
    4 ,5, 6};
  Matrix m(v,2,3);
  float correct[] = {
    1, 4,
    2, 5,
    3, 6};
  Matrix* list = new Matrix[]{{3,2}, {correct,3,2}};
  transpose<8>(m, list[0]);
  return list;
}

Matrix* getMulTest0() {
  float v0[] = {
    1, 3,
    2, 4};
  Matrix m0(v0,2,2);
  float v1[] = {
    2,
    0};
  Matrix m1(v1,2,1);
  float correct[]{
    2,
    4};
  Matrix* list = new Matrix[]{{2,1}, {correct,2,1}};
  matMul<8>(m0, m1, list[0]);
  return list;
} 

Matrix* getMulTest1() {
  float v0[] = { 
    1, 2, 0, 
    0 ,3, 0, 
    2, 1, 0};
  Matrix m0(v0,3,3);
  float v1[] = {  
    0, 2, 3, 
    0 ,0, 1, 
    2, 1, 0};
  Matrix m1(v1,3,3);
  float correct[] = {
    0, 2, 5, 
    0, 0, 3, 
    0, 4, 7};
  Matrix* list = new Matrix[]{{3,3}, {correct,3,3}};
  matMul<8>(m0, m1, list[0]);
  return list;
}

Matrix* getMulTest2() {
  float v0[] = { 
    0, 2, 3, 
    0 ,0, 0};
  Matrix m0(v0,2,3);
  float v1[] = { 
    1, 2, 
    0 ,3, 
    2, 1};
  Matrix m1(v1,3,2);
  float correct[] = {
    6, 9,
    0, 0,};
  Matrix * list = new Matrix[]{{2,2}, {correct,2,2}, {v0,2,3}, {v1,2,3}, };
  matMul<8>(m0, m1, list[0]);
  return list;
}

Matrix* getMulTest3() {
  float v0[] = { 
    2, 1, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 2, 0, 0, 0, 0, 0, 0,
    0, 1, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 1, 0, 2,
    0, 0, 0, 0, 0, 0, 0, 1, 0,};
  Matrix m0(v0,5,9);
  float v1[] = {  
    1, 1, 0, 
    0, 0, 1, 
    0, 1, 0, 
    0, 0, 0, 
    0, 0, 0, 
    0, 0, 0, 
    0, 1, 0, 
    1, 1, 0, 
    1, 0, 1,};
  Matrix m1(v1,9,3);
  float correct[] = {
    2, 2, 1, 
    0, 2, 0, 
    0, 0, 1, 
    2, 1, 2, 
    1, 1, 0,};
  Matrix* list = new Matrix[]{{5,3}, {correct,5,3}};
  matMul<8>(m0, m1, list[0]);
  return list;
}
Matrix* getMatMulScalerTest() {
  float v0[] = { 
    0, 2, 3, 
    0 ,0, 0};
  Matrix m0(v0,2,3);
  float v1[] = { 
    1, 2, 
    0 ,3, 
    2, 1};
  Matrix m1(v1,3,2);
  float correct[] = {
    -6, -9,
    0, 0,};
  Matrix * list = new Matrix[]{{2,2}, {correct,2,2}};
  //matMul<8, -1>(m0, m1, list[0]);
  return list;
}

Matrix* getMulTest0ATv() {
  float v0[] = {
    1, 3,
    2, 4};
  Matrix m0(v0,2,2);
  float v1[] = {
    2,
    0};
  Matrix m1(v1,2,1);
  float correct[]{
    2,
    6};
  Matrix* list = new Matrix[]{{2,1}, {correct,2,1}};
  matMulATv<8>(m0, m1, list[0]);
  return list;
} 

Matrix* getMulTest1ATv() {
  float v0[] = { 
    1, 0, 2, 
    2 ,3, 1, 
    0, 0, 0};
  Matrix m0(v0,3,3);
  float v1[] = {  
    3, 
    1, 
    0};
  Matrix m1(v1,3,1);
  float correct[] = {
    5, 
    3, 
    7};
  Matrix* list = new Matrix[]{{3,1}, {correct,3,1}};
  matMulATv<8>(m0, m1, list[0]);
  return list;
}

Matrix* getMulTest2ATv() {
  float v0[] = {
    1, 2, 4,
    3, 4, 2};
  Matrix m0(v0,2,3);
  float v1[] = {
    2,
    1};
  Matrix m1(v1,2,1);
  float correct[]{
    5,
    8,
    10};
  Matrix* list = new Matrix[]{{3,1}, {correct,3,1}};
  matMulATv<8>(m0, m1, list[0]);
  return list;
} 

Matrix* getMulTest0vvT() {
  float v0[] = { 
    3, 
    2};
  Matrix m0(v0,2,1);
  float v1[] = { 
    1, 
    0, 
    2};
  Matrix m1(v1,3,1);
  float correct[] = {
    3, 0, 6,
    2, 0, 4};
  Matrix * list = new Matrix[]{{2,3}, {correct,2,3}};
  matMulvvT<8>(m0, m1, list[0]);
  return list;
}

Matrix* getCorrelateTest0() {
  float v0[] = {  
    -1, 2, 
     1 ,0,

    -1, 2, 
     1 ,0,

    -1, 2, 
     1 ,0,

    -1, 2, 
     1 ,0,
  };
  Tensor<4>t0(v0,2,2,2,2);
  float v1[] = { 
    1, 2, 0, 
    0 ,3, 1, 
    2, 1, 0,

    1, 2, 0, 
    0 ,3, 1, 
    2, 1, 0,
  };
  Tensor<3>t1(v1,2,3,3);
  float correct[] = {
    6, 2, 
    16, 0,

    6, 2, 
    16, 0
  };
  Tensor<3> result(2,2,2);
  correlateAv<8>(t0, t1, result);
  Matrix* list = new Matrix[]{{result.v, 2,4}, {correct,2,4}};
  return list;
}

Matrix* getCorrelateTest1() {
  float v1[] = { 
    1, 2,  0,  3,
    0 ,3,  1,  1,
    2, 1,  0, -3,

    1, 2,  0,  3,
    0 ,3,  1,  1,
    2, 1,  0, -3,

    1, 2,  0,  3,
    0 ,3,  1,  1,
    2, 1,  0, -3,
  };
  Tensor<3>t1(v1,3,3,4);
  float v0[] = {  
    -1, 2, 
     1 ,0,

    -1, 2, 
     1 ,0,

    -1, 2, 
     1 ,0,

    -1, 2, 
     1 ,0,

    -1, 2, 
     1 ,0,

    -1, 2, 
     1 ,0,
  };
  Tensor<4>t0(v0,2,3,2,2);
  float correct[] = {
    9, 3, 21,
    24, 0, 3,

    9, 3, 21,
    24, 0, 3,
  };
  Tensor<3> result(2,2,3);
  correlateAv<8>(t0, t1, result);
  Matrix* list = new Matrix[]{{result.v,2,6}, {correct,2,6}};
  return list;
}

Matrix* getCorrelateTest2() {
  float v0[] = {  
    -1, 2, 
     1 ,0,

    -1, 2, 
     1 ,0,

    -1, 2, 
     1 ,0,
  };
  Tensor<3>t0(v0,3,2,2);
  float v1[] = { 
    1, 2,  0,  3,
    0 ,3,  1,  1,
    2, 1,  0, -3,

    1, 2,  0,  3,
    0 ,3,  1,  1,
    2, 1,  0, -3,
  };
  Tensor<3>t1(v1,2,3,4);
  float correct[] = {
    3, 1, 7,
    8, 0, 1,

    3, 1, 7,
    8, 0, 1,

    3, 1, 7,
    8, 0, 1,

    3, 1, 7,
    8, 0, 1,

    3, 1, 7,
    8, 0, 1,

    3, 1, 7,
    8, 0, 1,
  };
  Tensor<4> result(3,2,2,3);
  correlatevvT<8>(t0, t1, result);
  Matrix* list = new Matrix[]{{result.v,6,6}, {correct,6,6}};
  return list;
}

Matrix* getConvolveTest0() {
  float v0[] = {  
    1, 1, -1,
    1, 0, -1,
    1, 0, -1,

    1, 1, -1,
    1, 0, -1,
    1, 0, -1,

    1, 1, -1,
    1, 0, -1,
    1, 0, -1,

    1, 1, -1,
    1, 0, -1,
    1, 0, -1,

    1, 1, -1,
    1, 0, -1,
    1, 0, -1,

    1, 1, -1,
    1, 0, -1,
    1, 0, -1,

  };
  Tensor<4> t0(v0,3,2,3,3);
  float v1[] = { 
    2, 1, 4, 0,
    3, 2, 4, 2,
    0, 1, 0, 4,

    2, 1, 4, 0,
    3, 2, 4, 2,
    0, 1, 0, 4,

    2, 1, 4, 0,
    3, 2, 4, 2,
    0, 1, 0, 4,

  };
  Tensor<3> t1(v1,3,3,4);
  float correct[] = {
    6,  9,  9,  9, -12,  0,
    15,  18,  15,  9, -18, -6,
    15,  12,  12,  6, -12, -18,
    9,  9,  3,  9, -12, -18,
    0,  3,  0,  9,  0, -12,

    6,  9,  9,  9, -12,  0,
    15,  18,  15,  9, -18, -6,
    15,  12,  12,  6, -12, -18,
    9,  9,  3,  9, -12, -18,
    0,  3,  0,  9,  0, -12
  };
  Tensor<3> result(2,5,6);
  convolveATv<8>(t0, t1, result);
  return new Matrix[]{{result.v,5,6}, {correct,5,6}};
}

Matrix* getAddTest() {
  float v[] = { 
    1, 2, 2,
    0 ,3, 4, 
    2, 1, 5};
  Matrix m(v,3,3);
  float result[] = { 
    0, 2, 3, 
    0 ,0, 0,
    1, 2, 4};
  float correct[] = {
    1, 4, 5,
    0, 3, 4,
    3, 3, 9};
  Matrix * list = new Matrix[]{{result,3,3}, {correct,3,3}};
  addMat<8>(m, list[0]);
  return list;
}

int main(void) {
  MnistReader training_data("mnist/train-images-idx3-ubyte", "mnist/train-labels-idx1-ubyte");
  training_data.read_next();
  drawMat(training_data.last_read);
  std::cout << "\nsame image expected\n";
  MnistReader sub_reader = MnistReader(training_data, 0, 10);
  sub_reader.read_next();

  drawMat(sub_reader.last_read);

  testMatOperation(getTransposeTest0(), "transpose test 0");
  testMatOperation(getTransposeTest1(), "transpose test 1");
  testMatOperation(getTransposeTest2(), "transpose test 2");
  testMatOperation(getMulTest0(), "mulMat test 0");
  testMatOperation(getMulTest1(), "mulMat test 1");
  testMatOperation(getMulTest2(), "mulMat test 2");
  testMatOperation(getMulTest3(), "mulMat test 3");
  testMatOperation(getCorrelateTest0(), "correlate Av test 0");
  testMatOperation(getCorrelateTest1(), "correlate Av test 1");
  testMatOperation(getCorrelateTest2(), "correlate vvT test 0");
  testMatOperation(getConvolveTest0(), "convolve ATv test 0");
  testMatOperation(getAddTest(), "addMat test 0");
  testMatOperation(getMulTest0ATv(), "mulTestATv test 0");
  testMatOperation(getMulTest1ATv(), "mulTestATv test 1");
  testMatOperation(getMulTest2ATv(), "mulTestATv test 2");
  testMatOperation(getMulTest0vvT(), "mulTestvvT test0");

  TensorT<float, 2> tensor(1,2);
  std::cout << tensor.size << "\n"; 

}

