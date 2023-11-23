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

Matrix* getMulTest0ABT() {
  float v0[] = {
    1, 3,
    2, 4};
  Matrix m0(v0,2,2);
  float v1[] = {
    2,0};
  Matrix m1(v1,1,2);
  float correct[]{
    2,
    4};
  Matrix* list = new Matrix[]{{2,1}, {correct,2,1}, {v0,2,2}, {v1,1,2}};
  matMulATB<8>(m0, m1, list[0]);
  return list;
} 

Matrix* getMulTest1ABT() {
  float v0[] = { 
    0, 2, 3, 
    0 ,0, 0};
  Matrix m0(v0,2,3);
  float v1[] = { 
    1, 0, 2,
    2 ,3, 1};
  Matrix m1(v1,2,3);
  float correct[] = {
    6, 9,
    0, 0,};
  Matrix * list = new Matrix[]{{2,2}, {correct,2,2}};
  Matrix buffer = {3,2};
  transpose<8>(m1, buffer);
  matMul<8>(m0, buffer, list[0]);
  return list;
}

Matrix* getMulTest0ATB() {
  float v0[] = {
    1, 2,
    3, 4};
  Matrix m0(v0,2,2);
  float v1[] = {
    2,
    0};
  Matrix m1(v1,2,1);
  float correct[]{
    2,
    4};
  float result[]{
    2,
    -1};
  Matrix* list = new Matrix[]{{result, 2,1}, {correct,2,1}};
  matMulATB<8>(m0, m1, list[0]);
  return list;
} 

Matrix* getMulTest1ATB() {
  float v0[] = { 
    1, 0, 2, 
    2 ,3, 1, 
    0, 0, 0};
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
  matMulATB<8>(m0, m1, list[0]);
  return list;
}

Matrix* getMulTest2ATB() {
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
  matMulATB<8>(m0, m1, list[0]);
  return list;
} 

Matrix* getCorrelateTest0() {
  float v0[] = { 
    1, 2, 0, 
    0 ,3, 1, 
    2, 1, 0};
  Matrix m0(v0,3,3);
  float v1[] = {  
    -1, 2, 
    1 ,0};
  Matrix m1(v1,2,2);
  float correct[] = {
    3, 1, 
    8, 0};
  Matrix* list = new Matrix[]{{2,2}, {correct,2,2}};
  correlate<8>(m0, m1, list[0]);
  return list;
}

Matrix* getCorrelateTest1() {
  float v0[] = { 
    1, 2,  0,  3,
    0 ,3,  1,  1,
    2, 1,  0, -3};
  Matrix m0(v0,3,4);
  float v1[] = {  
    -1, 2, 
    1 ,0};
  Matrix m1(v1,2,2);
  float correct[] = {
    3, 1, 7,
    8, 0, 1};
  Matrix* list = new Matrix[]{{2,3}, {correct,2,3}};
  correlate<8>(m0, m1, list[0]);
  return list;
}

Matrix* getConvolveTest0() {
  float v0[] = { 
    2, 1, 4, 0,
    3, 2, 4, 2,
    0, 1, 0, 4
  };
  Matrix m0(v0,3,4);
  float v1[] = {  
    1, 1, -1,
    1, 0, -1,
    1, 0, -1
  };
  Matrix m1(v1,3,3);
  float correct[] = {
    2,  3,  3,  3, -4,  0,
    5,  6,  5,  3, -6, -2,
    5,  4,  4,  2, -4, -6,
    3,  3,  1,  3, -4, -6,
    0,  1,  0,  3,  0, -4
  };
  Matrix* list = new Matrix[]{{5,6}, {correct,5,6}};
  convolveFull<8>(m0, m1, list[0]);
  return list;
}

Matrix* getConvolveTest1() {
  float v0[] = {  
    1, 1, -1,
    1, 0, -1,
    1, 0, -1
  };
  Matrix m0(v0,3,3);
  float v1[] = { 
    2, 1, 4, 0,
    3, 2, 4, 2,
    0, 1, 0, 4
  };
  Matrix m1(v1,3,4);
  float correct[] = {
    2,  3,  3,  3, -4,  0,
    5,  6,  5,  3, -6, -2,
    5,  4,  4,  2, -4, -6,
    3,  3,  1,  3, -4, -6,
    0,  1,  0,  3,  0, -4
  };
  Matrix* list = new Matrix[]{{5,6}, {correct,5,6}};
  convolveFull<8>(m0, m1, list[0]);
  return list;
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

Matrix* gettest() {
  size_t size = 1024; 
  Matrix* list = new Matrix[]{
    {size, size}, 
    {size, size}, 
    {size, size}};
  randomizeMat(list[0]);
  randomizeMat(list[1]);
  return list;
}

int main(void) {
  //MnistReader training_data("mnist/train-images-idx3-ubyte", "mnist/train-labels-idx1-ubyte");
  //training_data.read_next();
  //drawMat(training_data.last_read);
  //std::cout << "\nsame image expected\n";
  //MnistReader sub_reader = MnistReader(training_data, 0, 10);
  //sub_reader.read_next();

  //drawMat(sub_reader.last_read);
  testMatOperation(getTransposeTest0(), "transpose test 0");
  testMatOperation(getTransposeTest1(), "transpose test 1");
  testMatOperation(getTransposeTest2(), "transpose test 2");
  testMatOperation(getMulTest0(), "mulMat test 0");
  testMatOperation(getMulTest1(), "mulMat test 1");
  testMatOperation(getMulTest2(), "mulMat test 2");
  testMatOperation(getMulTest3(), "mulMat test 3");
  testMatOperation(getCorrelateTest0(), "correlate test 0");
  testMatOperation(getCorrelateTest1(), "correlate test 1");
  testMatOperation(getConvolveTest0(), "convolve test 0");
  testMatOperation(getConvolveTest1(), "convolve test 1");
  testMatOperation(getAddTest(), "addMat test 0");
  testMatOperation(getMulTest1ABT(), "mulTestABT test 1");
  testMatOperation(getMulTest0ATB(), "mulTestATB test 0");
  testMatOperation(getMulTest1ATB(), "mulTestATB test 1");
  testMatOperation(getMulTest2ATB(), "mulTestATB test 2");

  Tensor<float, 2> tensor(1,2);
  std::cout << tensor.size << "\n"; 

}

