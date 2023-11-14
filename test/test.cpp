#include <algorithm>
#include <algorithm>
#include <assert.h>
#include <cstddef>
#include <cstdio>
#include <iostream>
#include "../src/network/math.hpp"
#include "../src/network/net.hpp"
#include "../src/mnist_reader.hpp"

#define TEST(x, error_message) { if (!(x)) std::cout << __FUNCTION__ << " failed on line " << __LINE__  << " executing " << error_message << std::endl; }

Matrix* getTransposeTest0() {
  float v[] = { 
    0, 2, 3, 
    0 ,0, 0,
    1, 2, 4};
  Matrix m = Matrix(3,3,v);
  float correct[] = {
    0, 0, 1,
    2, 0, 2,
    3, 0, 4};
  Matrix *list = new Matrix[]{{3,3}, {3,3,correct}};
  transpose<8>(m, list[0]);
  return list;
}

Matrix* getTransposeTest1() {
  float v[] = { 
    1, 
    2,
    3};
  Matrix m = Matrix(3,1,v);
  float* correct = {new float[]{
    1, 2, 3}};
  Matrix *list = new Matrix[]{{1,3}, {1,3,correct}};
  transpose<8>(m, list[0]);
  return list;
}

Matrix* getTransposeTest2() {
  float v[] = { 
    1, 2, 3,
    4 ,5, 6};
  Matrix m = {2,3,v};
  float correct[] = {
    1, 4,
    2, 5,
    3, 6};
  Matrix* list = new Matrix[]{{3,2}, {3,2,correct}};
  transpose<8>(m, list[0]);
  return list;
}

Matrix* getMulTest0() {
  float v0[] = {
    1, 3,
    2, 4};
  Matrix m0 = {2,2,v0};
  float v1[] = {
    2,
    0};
  Matrix m1 = {2,1,v1};
  float correct[]{
    2,
    4};
  Matrix* list = new Matrix[]{{2,1}, {2,1,correct}};
  matMul<8>(m0, m1, list[0]);
  return list;
} 

Matrix* getMulTest1() {
  float v0[] = { 
    1, 2, 0, 
    0 ,3, 0, 
    2, 1, 0};
  Matrix m0 = {3,3,v0};
  float v1[] = {  
    0, 2, 3, 
    0 ,0, 1, 
    2, 1, 0};
  Matrix m1 = {3,3,v1};
  float correct[] = {
    0, 2, 5, 
    0, 0, 3, 
    0, 4, 7};
  Matrix* list = new Matrix[]{{3,3}, {3,3,correct}};
  matMul<8>(m0, m1, list[0]);
  return list;
}

Matrix* getMulTest2() {
  float v0[] = { 
    0, 2, 3, 
    0 ,0, 0};
  Matrix m0 = {2,3,v0};
  float v1[] = { 
    1, 2, 
    0 ,3, 
    2, 1};
  Matrix m1 = {3,2,v1};
  float correct[] = {
    6, 9,
    0, 0,};
  Matrix * list = new Matrix[]{{2,2}, {2,2,correct}, {2,3,v0}, {3,2,v1}, };
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
  Matrix m0 = {5,9,v0};
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
  Matrix m1 = {9,3,v1};
  float correct[] = {
    2, 2, 1, 
    0, 2, 0, 
    0, 0, 1, 
    2, 1, 2, 
    1, 1, 0,};
  Matrix* list = new Matrix[]{{5,3}, {5,3,correct}};
  matMul<8>(m0, m1, list[0]);
  return list;
}
Matrix* getMatMulScalerTest() {
  float v0[] = { 
    0, 2, 3, 
    0 ,0, 0};
  Matrix m0 = {2,3,v0};
  float v1[] = { 
    1, 2, 
    0 ,3, 
    2, 1};
  Matrix m1 = {3,2,v1};
  float correct[] = {
    -6, -9,
    0, 0,};
  Matrix * list = new Matrix[]{{2,2}, {2,2,correct}};
  matMul<8, -1>(m0, m1, list[0]);
  return list;
}

Matrix* getMulTest0ABT() {
  float v0[] = {
    1, 3,
    2, 4};
  Matrix m0 = {2,2,v0};
  float v1[] = {
    2,0};
  Matrix m1 = {1,2,v1};
  float correct[]{
    2,
    4};
  Matrix* list = new Matrix[]{{2,1}, {2,1,correct}, {2,2,v0}, {1,2,v1}};
  matMulATB<8>(m0, m1, list[0]);
  return list;
} 

Matrix* getMulTest1ABT() {
  float v0[] = { 
    0, 2, 3, 
    0 ,0, 0};
  Matrix m0 = {2,3,v0};
  float v1[] = { 
    1, 0, 2,
    2 ,3, 1};
  Matrix m1 = {2,3,v1};
  float correct[] = {
    6, 9,
    0, 0,};
  Matrix * list = new Matrix[]{{2,2}, {2,2,correct}};
  Matrix buffer = {3,2};
  transpose<8>(m1, buffer);
  matMul<8>(m0, buffer, list[0]);
  return list;
}

Matrix* getMulTest0ATB() {
  float v0[] = {
    1, 2,
    3, 4};
  Matrix m0 = {2,2,v0};
  float v1[] = {
    2,
    0};
  Matrix m1 = {2,1,v1};
  float correct[]{
    2,
    4};
  float result[]{
    2,
    -1};
  Matrix* list = new Matrix[]{{2,1, result}, {2,1,correct}};
  matMulATB<8>(m0, m1, list[0]);
  return list;
} 

Matrix* getMulTest1ATB() {
  float v0[] = { 
    1, 0, 2, 
    2 ,3, 1, 
    0, 0, 0};
  Matrix m0 = {3,3,v0};
  float v1[] = {  
    0, 2, 3, 
    0 ,0, 1, 
    2, 1, 0};
  Matrix m1 = {3,3,v1};
  float correct[] = {
    0, 2, 5, 
    0, 0, 3, 
    0, 4, 7};
  Matrix* list = new Matrix[]{{3,3}, {3,3,correct}};
  matMulATB<8>(m0, m1, list[0]);
  return list;
}

Matrix* getAddTest() {
  float v[] = { 
    1, 2, 2,
    0 ,3, 4, 
    2, 1, 5};
  Matrix m = {3,3,v};
  float result[] = { 
    0, 2, 3, 
    0 ,0, 0,
    1, 2, 4};
  float correct[] = {
    1, 4, 5,
    0, 3, 4,
    3, 3, 9};
  Matrix * list = new Matrix[]{{3,3,result}, {3,3,correct}};
  addMat<8>(m, list[0]);
  return list;
}

void testMatOperation(Matrix* list, std::string text) {
  for(size_t x = 0; x < list[1].ht; x++) {
    for(size_t y = 0; y < list[1].wt; y++) {
      if(list[0][x][y] != list[1][x][y]) {
        std::cout << "\ngot: \n";
        printMat(list[0]);
        std::cout << "expected: \n";
        printMat(list[1]);

        std::string error_message = text + " with indices " + std::to_string(x) +" "+ std::to_string(y);
        TEST(false, error_message);
        return;
      }
    }
  }
  
  delete [] list;
  std::cout << text << " complete\n";
}

int main(void) {
  MnistReader training_data("mnist/train-images-idx3-ubyte", "mnist/train-labels-idx1-ubyte");
  training_data.read_next();
  training_data.read_next();
  drawMat(training_data.last_read);
  std::cout << "\nsame image expected\n\n";
  MnistReader sub_reader = MnistReader(training_data, 0, 10);
  sub_reader.read_next();
  sub_reader.read_next();
  drawMat(sub_reader.last_read);
  std::cout << "\n";

  testMatOperation(getTransposeTest0(), "transpose test 0");
  testMatOperation(getTransposeTest1(), "transpose test 1");
  testMatOperation(getTransposeTest2(), "transpose test 2");
  testMatOperation(getMulTest0(), "mulMat test 0");
  testMatOperation(getMulTest1(), "mulMat test 1");
  testMatOperation(getMulTest2(), "mulMat test 2");
  testMatOperation(getMulTest3(), "mulMat test 3");
  testMatOperation(getMatMulScalerTest(), "scaler test");
  testMatOperation(getAddTest(), "addMat test 0");
  testMatOperation(getMulTest1ABT(), "mulTestABT test 1");
  testMatOperation(getMulTest0ATB(), "mulTestATB test 0");
  testMatOperation(getMulTest1ATB(), "mulTestATB test 1");
}
