#include <algorithm>
#include <algorithm>
#include <assert.h>
#include <cstddef>
#include <cstdio>
#include <cstring>
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
  Vector m1(v1,2);
  float correct[]{
    2,
    6};
  Matrix* list = new Matrix[]{{2,1}, {correct,2,1}};
  Vector result = list[0].vectorize();
  matMulATv(m0, m1, result);
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
  Vector m1(v1,3);
  float correct[] = {
    5, 
    3, 
    7};
  Matrix* list = new Matrix[]{{3,1}, {correct,3,1}};
  Vector result = list[0].vectorize();
  matMulATv(m0, m1, result);
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
  Vector m1(v1,2);
  float correct[]{
    5,
    8,
    10};
  Matrix* list = new Matrix[]{{3,1}, {correct,3,1}};
  Vector result = list[0].vectorize();
  matMulATv(m0, m1, result);
  return list;
} 

Matrix* getMulTest0vvT() {
  float v0[] = { 
    3, 
    2};
  Vector m0(v0,2);
  float v1[] = { 
    1, 
    0, 
    2};
  Vector m1(v1,3);
  float correct[] = {
    3, 0, 6,
    2, 0, 4};
  Matrix * list = new Matrix[]{{2,3}, {correct,2,3}};
  matMulvvT(m0, m1, list[0]);
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
  correlateAv(t0, t1, result);
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
  correlateAv(t0, t1, result);
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
  correlatevvT(t0, t1, result);
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
  convolveATv(t0, t1, result);
  return new Matrix[]{{result.v,5,6}, {correct,5,6}};
}

Matrix* getAddTest() {
  float v[] = { 
    1, 2, 2, 1, 2, 2, 1, 2, 2,
    0 ,3, 4, 0 ,3, 4, 0 ,3, 4, 
    2, 1, 5, 2, 1, 5, 2, 1, 5
  };
  Matrix m(v,3,3);
  float result[] = { 
    0, 2, 3, 0, 2, 3, 0, 2, 3, 
    0 ,0, 0, 0 ,0, 0, 0 ,0, 0,
    1, 2, 4, 1, 2, 4, 1, 2, 4,
  };
  float correct[] = {
    1, 4, 5, 1, 4, 5, 1, 4, 5,
    0, 3, 4, 0, 3, 4, 0, 3, 4,
    3, 3, 9, 3, 3, 9, 3, 3, 9,
  };
  Matrix * list = new Matrix[]{{result,3,3}, {correct,3,3}};
  addTens(m, list[0]);
  return list;
}

Matrix* getMaxpoolingTest0() {
  float v[] = {
    1,2,3,4,6,
    2,3,4,5,5,
    5,6,7,8,4,
    7,8,9,0,5,
    3,1,6,3,6,
             
    1,2,3,4,6,
    2,3,4,5,5,
    5,6,7,8,4,
    7,8,9,0,5,
    3,1,6,3,6,
             
    1,2,3,4,6,
    2,3,4,5,5,
    5,6,7,8,4,
    7,8,9,0,5,
    3,1,6,3,6,
  };
  Tensor<3> t(v, 3,5,5);
  Tensor<3> result(3,3,3);
  float correct[] = {
    3,5,6,
    8,9,5,
    3,6,6,

    3,5,6,
    8,9,5,
    3,6,6,

    3,5,6,
    8,9,5,
    3,6,6,
  };
  Shape<2> kernel(2,2);
  TensorT<size_t, 3> max_locations(3,3,3);
  maxPooling(t, kernel, result, max_locations);
  Matrix * list = new Matrix[]{{result.v,3,9}, {correct,3,9}};

  //maxPooling_backward(result, t, max_locations);
  //printMat(t[0]);
  //std::cout << "\n";
  //printMat(t[1]);
  //std::cout << "\n";
  //printMat(t[2]);
  //std::cout << "\n";

  return list;
}

Matrix* getMaxpoolingTest1() {
  float v[] = {
    1,2,3,4,
    2,3,4,5,
    5,6,7,8,
    7,8,9,0,

    1,2,3,4,
    2,3,4,5,
    5,6,7,8,
    7,8,9,0,

    1,2,3,4,
    2,3,4,5,
    5,6,7,8,
    7,8,9,0,
  };
  Tensor<3> t(v, 3,4,4);
  Tensor<3> result(3,2,2);
  float correct[] = {
    5,7,
    13,14,

    21,23,
    29,30,

    37,39,
    45,46,
  };
  Shape<2> kernel(2,2);
  TensorT<size_t, 3> max_locations(3,2,2);
  Tensor<3> res(3,2,2);
  maxPooling(t, kernel, result, max_locations);
  for(size_t i = 0; i < res.size; i++) res.v[i] = max_locations.v[i];
  Matrix * list = new Matrix[]{{res.v ,3,4}, {correct ,3,4}};

 return list;
}

Matrix* getMaxpoolingTest2() {
  float v[] = {
    1,2,3,4,9,
    2,3,4,5,9,
    5,6,7,8,9,
    7,8,9,0,9,
    9,9,9,9,9,
             
    -7,-7,3,4,9,
    -5,-3,4,5,9,
    5,6,7,8,9,
    7,8,9,0,9,
    9,9,9,9,9,
             
    1,2,3,4,9,
    2,3,4,5,9,
    5,6,7,8,9,
    7,8,9,0,9,
    9,9,9,9,9,
  };
  Tensor<3> t(v, 3,5,5);
  Tensor<3> result(3,5,5);
  Tensor<3> correct(t);
  Shape<2> kernel(1,1);
  TensorT<size_t, 3> max_locations(3,5,5);
  maxPooling(t, kernel, result, max_locations);
  Matrix * list = new Matrix[]{{result.v,3,25}, {correct.v,3,25}};

  //maxPooling_backward(result, t, max_locations);
  //printMat(t[0]);
  //std::cout << "\n";
  //printMat(t[1]);
  //std::cout << "\n";
  //printMat(t[2]);
  //std::cout << "\n";

  //draw3D(t);

  return list;
}

int main(void) {
  MnistReader training_set("data/fashion_mnist_train_vectors.csv", "data/fashion_mnist_train_labels.csv", {28,28}, 60000);
  MnistReader sub_reader = MnistReader(training_set, 20, 30);
  draw3D(sub_reader.getAllImages());
  std::cout << "\nsame images expected\n\n";
  draw3D(training_set.getAllImages(), 20);

  Tensor<3> result(1,14,14), input(1,28,28);
  TensorT<size_t, 3> loc(1,14,14);
  Shape<2> kernel(2,2);
  copyToTensorOfSameSize(sub_reader.getAllImages()[5], input);

  maxPooling(input, kernel, result, loc);

  std::cout << "\nmax pooling\n";
  draw3D(input);
  std::cout << "\n";
  draw3D(result);
  maxPooling_backward(result, input, loc);

  std::cout << "\nmax pooling backward\n";
  draw3D(input);
  std::cout << "\n";

  //draw3D(training_set.images);
  //training_set.augment({-1,-1});
  //std::cout << "\nAugmentation one up one left\n\n";
  //draw3D(training_set.augmented_images);
  //training_set.loop_to_beg();
  //std::cout << "\n";
  
  training_set.loop_to_beg();
  drawMat(training_set.getAllImages()[training_set.permutation[0]]);
  std::cout << "\nAugmentation\n\n";
  Matrix augmented = training_set.getAllImages()[training_set.permutation[0]];
  size_t y;
  training_set.read_next(false, augmented, y);
  drawMat(augmented);

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
  testMatOperation(getMaxpoolingTest1(), "maxpooling test indicies");
  testMatOperation(getMaxpoolingTest0(), "maxpooling test");
  //testMatOperation(getMaxpoolingTest2(), "maxpooling test kernel size 1 ");

}

