#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cstdio>
#include <iostream>
#include <chrono>
#include <string>
#include "../src/network/math.hpp"
#include "../src/network/layer.hpp"
#include "benchmark.hpp"

Matrix* getMulTest0() {
  size_t size1 = 1024; 
  size_t size2 = 1024;
  Matrix* list = new Matrix[]{
    {size2, size1}, 
    {size1, 1}, 
    {size2, 1}};
  randomize(list[0]);
  randomize(list[1]);
  return list;
}

Matrix* getMulTest1() {
  size_t size1 = 2000; 
  size_t size2 = 12000;
  Matrix* list = new Matrix[]{
    {size2, size1}, 
    {size1, 1}, 
    {size2, 1}};
  randomize(list[0]);
  randomize(list[1]);
  return list;
}

Matrix* getMulTest2() {
  size_t wt = 2048; 
  size_t ht = 512; 
  Matrix* list = new Matrix[]{
    {ht, wt}, 
    {wt, ht}, 
    {wt, ht}};
  randomize(list[0]);
  randomize(list[1]);
  return list;
}

Matrix* getMulTest3() {
  size_t size = 1024; 
  Matrix* list = new Matrix[]{
    {size, size}, 
    {size, size}, 
    {size, size}};
  randomize(list[0]);
  randomize(list[1]);
  return list;
}
void benchMatMul() {
  Matrix* list = getMulTest0();
  Timer timer = Timer("bench matMul 8: %0.3f ms \n");
  matMul<8>(list[0], list[1], list[2]);
}

void benchMatMul1() {
  Matrix* list = getMulTest0();
  Timer timer = Timer("bench matMul 16: %0.3f ms \n");
  matMul<16>(list[0], list[1], list[2]);
}

void benchMatMul2() {
  Matrix* list = getMulTest0();
  Timer timer = Timer("bench matMul 32: %0.3f ms \n");
  matMul<32>(list[0], list[1], list[2]);
}

void benchMatMul3() {
  Matrix* list = getMulTest0();
  Timer timer = Timer("bench matMul 64: %0.3f ms \n");
  matMul<64>(list[0], list[1], list[2]);
}

void benchMatMul4() {
  Matrix* list = getMulTest0();
  Timer timer = Timer("bench matMul 128: %0.3f ms \n");
  matMul<128>(list[0], list[1], list[2]);
}

void benchCorrelate() {
  Tensor<4> kernels(5,1,3,3);
  Tensor<3> input(1,1024, 1024);
  Tensor<3> result(5,1024-2,1024-2);
  randomize(kernels);
  randomize(input);

  Timer timer = Timer("bench correlate: %0.3f ms \n");
  correlateAv(kernels, input, result);
  correlateAv(kernels, input, result);
  correlateAv(kernels, input, result);
  correlateAv(kernels, input, result);
  correlateAv(kernels, input, result);
  correlateAv(kernels, input, result);
  correlateAv(kernels, input, result);
  correlateAv(kernels, input, result);
}

void benchConv() {
  Tensor<4> kernels(5,1,3,3);
  Tensor<3> input(5,1024-2,1024-2);
  Tensor<3> result(1,1024, 1024);
  randomize(kernels);
  randomize(input);

  Timer timer = Timer("bench convolve: %0.3f ms \n");
  convolveATv(kernels, input, result);
  convolveATv(kernels, input, result);
  convolveATv(kernels, input, result);
  convolveATv(kernels, input, result);
  convolveATv(kernels, input, result);
  convolveATv(kernels, input, result);
  convolveATv(kernels, input, result);
  convolveATv(kernels, input, result);
}


void benchMatMulScaler1() {
  Matrix* list = getMulTest1();
  Timer timer = Timer("bench matMul with scaler 1: %0.3f ms \n");
  matMul<8, 1>(list[0], list[1], list[2]);
}

void benchMatMulABT() {
  Matrix* list = getMulTest1();
  Timer timer = Timer("bench matMulABT: %0.3f ms \n");
  matMul<8>(list[0], list[1], list[2]);
}

void benchMatMulATv() {
  Matrix* list = getMulTest1();
  Vector v = list[2].vectorize();
  Vector result = list[1].vectorize();
  Timer timer = Timer("bench matMulATv: %0.3f ms \n");
  matMulATv(list[0], v, result);
}

void benchTranspose0() {
  Matrix* list = getMulTest3();
  Timer timer = Timer("bench transpose square matrix: %0.3f ms \n");
  transpose<8>(list[0], list[2]);
}

void benchTranspose1() {
  Matrix* list = getMulTest2();
  Timer timer = Timer("bench transpose rectangular matrix wider: %0.3f ms \n");
  transpose<8>(list[0], list[2]);
}

void benchTranspose2() {
  Matrix* list = getMulTest2();
  Timer timer = Timer("bench transpose rectangular matrix taller: %0.3f ms \n");
  transpose<8>(list[1], list[0]);
}

void benchAddMat() {
  Matrix* list = getMulTest3();
  Timer timer = Timer("bench add matricies: %0.3f ms \n");
  addTens(list[1], list[0]);
}

void benchZeroMat() {
  Matrix* list = getMulTest3();
  Timer timer = Timer("bench zero mat: %0.3f ms \n");
  zero(list[0]);
  timer.stop();
}

void benchMaxPooling() {
  Tensor<3> input(32,28,28);
  Tensor<3> result(64,13,13);
  TensorT<size_t , 3> loc(64,13,13);
  Shape<2> kernel(2,2);
  randomize(input);

  Timer timer = Timer("bench max pooling: %0.3f ms \n");
  maxPooling(input, kernel, result, loc);
}

int main(void) {
  std::cout << "find optimal tailing\n";
  benchMatMul();
  benchMatMul1();
  benchMatMul2();
  benchMatMul3();
  benchMatMul4();

  std::cout << "\n";
  benchMatMulScaler1();
  benchMatMulABT();
  benchMatMulATv();

  std::cout << "\n";
  benchCorrelate();
  benchConv();

  std::cout << "\n";
  benchTranspose0();
  benchTranspose1();
  benchTranspose2();

  std::cout << "\n";
  benchAddMat();
  benchZeroMat();

  std::cout << "\n";
  benchMaxPooling();
}
