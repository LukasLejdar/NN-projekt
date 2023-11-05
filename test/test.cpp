#include <algorithm>
#include <algorithm>
#include <assert.h>
#include <cstdio>
#include <iostream>
#include "../src/network/math.hpp"

#define TEST(x, error_message) { if (!(x)) std::cout << __FUNCTION__ << " failed on line " << __LINE__  << " executing " << error_message << std::endl; }

Matrix* getTransposeTest0() {
  float* v = {new float[]{0}};
  float* v0 = {new float[]{ 
    0, 2, 3, 
    0 ,0, 0,
    1, 2, 4}};
  float* correct = {new float[]{
    0, 0, 1,
    2, 0, 2,
    3, 0, 4}};
  float* result = new float[9];
  std::fill(result, result+9, 0);
  Matrix *list = new Matrix[]{{3,3,result}, {3,3,correct}, {3,3,v0}, {1,1,v}};
  transpose<3,3,8>(list[2], list[0]);
  return list;
}

Matrix* getTransposeTest1() {
  float* v = {new float[]{0}};
  float* v0 = {new float[]{ 
    1, 
    2,
    3}};
  float* correct = {new float[]{
    1, 2, 3}};
  float* result = new float[6];
  std::fill(result, result+6, 0);
  Matrix *list = new Matrix[]{{1,3,result}, {1,3,correct}, {3,1,v0}, {1,1,v}};
  transpose<3,1,8>(list[2], list[0]);
  return list;
}

Matrix* getTransposeTest2() {
  //  1, 2, 3, 4 ,5, 6
  //  1, 4, 2, 5, 3, 6
  float* v = {new float[]{0}};
  float* v0 = {new float[]{ 
    1, 2, 3,
    4 ,5, 6}};
  float* correct = {new float[]{
    1, 4,
    2, 5,
    3, 6}};
  float* result = new float[6];
  std::fill(result, result+6, 0);
  Matrix *list = new Matrix[]{{3,2,result}, {3,2,correct}, {2,3,v0}, {1,1,v}};
  transpose<2,3,8>(list[2], list[0]);
  return list;
}

Matrix* getMulTest0() {
  float* v0 = {new float[]{
    1, 3,
    2, 4}};
  float* v1 = {new float[]{
    2,
    0}};
  float* correct{new float[]{
    2,
    4}};
  float* result = new float[2];
  std::fill(result, result+2, 0);
  Matrix* list = new Matrix[]{{2,1,result}, {2,1,correct}, {2,2,v0}, {2,1,v1}};
  addMulMat<2,2,1,8>(list[2], list[3], list[0]);
  return list;
} 

Matrix* getMulTest1() {
  float* v0 = {new float[]{ 
    1, 2, 0, 
    0 ,3, 0, 
    2, 1, 0}};
  float* v1 = {new float[]{  
    0, 2, 3, 
    0 ,0, 1, 
    2, 1, 0}};
  float* correct = {new float[]{
    0, 2, 5, 
    0, 0, 3, 
    0, 4, 7}};
  float* result = new float[9];
  std::fill(result, result+9, 0);
  Matrix* list = new Matrix[]{{3,3,result}, {3,3,correct}, {3,3,v0}, {3,3,v1}};
  addMulMat<3,3,3,8>(list[2], list[3], list[0]);
  return list;
}

Matrix* getMulTest2() {
  float* v0 = {new float[]{ 
    0, 2, 3, 
    0 ,0, 0}};
  float* v1 = {new float[]{ 
    1, 2, 
    0 ,3, 
    2, 1}};
  float* correct = {new float[]{
    6, 9,
    0, 0,}};
  float* result = new float[4]; 
  std::fill(result, result+4, 0);
  Matrix * list = new Matrix[]{{2,2,result}, {2,2,correct}, {2,3,v0}, {3,2,v1}, };
  addMulMat<2,3,2,8>(list[2], list[3], list[0]);
  return list;
}

Matrix* getMulTest0ABT() {
  float* v0 = {new float[]{
    1, 3,
    2, 4}};
  float* v1 = {new float[]{
    2,0}};
  float* correct{new float[]{
    2,
    4}};
  float* result = new float[2];
  std::fill(result, result+2, 0);
  Matrix* list = new Matrix[]{{2,1,result}, {2,1,correct}, {2,2,v0}, {1,2,v1}};
  addMulMatATB<2,2,1,8>(list[2], list[3], list[0]);
  return list;
} 

Matrix* getMulTest1ABT() {
  float* v0 = {new float[]{ 
    0, 2, 3, 
    0 ,0, 0}};
  float* v1 = {new float[]{ 
    1, 0, 2,
    2 ,3, 1}};
  float* correct = {new float[]{
    6, 9,
    0, 0,}};
  float* result = new float[4]; 
  std::fill(result, result+4, 0);
  Matrix * list = new Matrix[]{{2,2,result}, {2,2,correct}, {2,3,v0}, {2,3,v1}};

  float buffer[] = {0,0,0,0,0,0};
  Matrix mb = {3,2,buffer};
  transpose<2,3,8>(list[3], mb);
  addMulMat<2,3,2,8>(list[2], mb, list[0]);
  return list;
}

void delete_test_data(Matrix list[]) {
  for(int i = 0; i < 4; i++) { delete list[i].v; }
  delete[] list;
}

Matrix* getMulTest0ATB() {
  float* v0 = {new float[]{
    1, 2,
    3, 4}};
  float* v1 = {new float[]{
    2,
    0}};
  float* correct{new float[]{
    2,
    4}};
  float* result = new float[2];
  std::fill(result, result+2, 0);
  Matrix* list = new Matrix[]{{2,1,result}, {2,1,correct}, {2,2,v0}, {2,1,v1}};
  addMulMatATB<2,2,1,8>(list[2], list[3], list[0]);
  return list;
} 

Matrix* getMulTest1ATB() {
  float* v0 = {new float[]{ 
    1, 0, 2, 
    2 ,3, 1, 
    0, 0, 0}};
  float* v1 = {new float[]{  
    0, 2, 3, 
    0 ,0, 1, 
    2, 1, 0}};
  float* correct = {new float[]{
    0, 2, 5, 
    0, 0, 3, 
    0, 4, 7}};
  float* result = new float[9];
  std::fill(result, result+9, 0);
  Matrix* list = new Matrix[]{{3,3,result}, {3,3,correct}, {3,3,v0}, {3,3,v1}};
  addMulMatATB<3,3,3,8>(list[2], list[3], list[0]);
  return list;
}

Matrix* getAddTest() {
  float* v = {new float[]{0}};
  float* v0 = {new float[]{ 
    1, 2, 2,
    0 ,3, 4, 
    2, 1, 5}};
  float* result = {new float[]{ 
    0, 2, 3, 
    0 ,0, 0,
    1, 2, 4}};
  float* correct = {new float[]{
    1, 4, 5,
    0, 3, 4,
    3, 3, 9}};
  Matrix * list = new Matrix[]{{3,3,result}, {3,3,correct}, {3,3,v0}, {1,1,v}};
  addMat<3,3,8>(list[2], list[0]);
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

  delete_test_data(list);
  std::cout << text << " complete\n";
}

int main(void) {
  testMatOperation(getTransposeTest0(), "transpose test 0");
  testMatOperation(getTransposeTest1(), "transpose test 1");
  testMatOperation(getTransposeTest2(), "transpose test 2");
  testMatOperation(getMulTest0(), "mulMat test 0");
  testMatOperation(getMulTest1(), "mulMat test 1");
  testMatOperation(getMulTest2(), "mulMat test 2");
  testMatOperation(getAddTest(), "addMat test 0");
  //testMatOperation(getMulTest0ABT(), "mulTestABT test 0");
  testMatOperation(getMulTest1ABT(), "mulTestABT test 1");
  testMatOperation(getMulTest0ATB(), "mulTestATB test 0");
  testMatOperation(getMulTest1ATB(), "mulTestATB test 1");
}















