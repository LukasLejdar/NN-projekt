#include <assert.h>
#include <cstdio>
#include <iostream>
#include "../src/network/math.hpp"

#define TEST(x, error_message) { if (!(x)) std::cout << __FUNCTION__ << " failed on line " << __LINE__  << " executing " << error_message << std::endl; }

Matrix* getMulTest0() {
  float* v0{new float[]{
    1, 3,
    2, 4}};
  float* v1{new float[]{
    2,
    0}};
  float* correct{new float[]{
    2,
    4}};
  float* result = new float[2];
  Matrix* list = new Matrix[]{{2,2,v0}, {2,1,v1}, {2,1,result}, {2,1,correct}};
  mulMat(list[0], list[1], list[2]);
  return list;
} 

Matrix* getMulTest1() {
  float* v0{new float[]{ 
    1, 2, 0, 
    0 ,3, 0, 
    2, 1, 0}};
  float* v1{new float[]{  
    0, 2, 3, 
    0 ,0, 1, 
    2, 1, 0}};
  float* correct = {new float[]{
    0, 2, 5, 
    0, 0, 3, 
    0, 4, 7}};
  float* result = new float[9];
  Matrix* list = new Matrix[]{{3,3,v0}, {3,3,v1}, {3,3,result}, {3,3,correct}};
  mulMat(list[0], list[1], list[2]);
  return list;
}

Matrix* getMulTest2() {
  float* v0{new float[]{ 
    0, 2, 3, 
    0 ,0, 0}};
  float* v1{new float[]{ 
    1, 2, 
    0 ,3, 
    2, 1}};
  float* correct = {new float[]{
    6, 9,
    0, 0,}};
  float* result = new float[4]; 
  Matrix * list = new Matrix[]{{2,3,v0}, {3,2,v1}, {2,2,result}, {2,2,correct}};
  mulMat(list[0], list[1], list[2]);
  return list;
}

Matrix* getMulTest0ABT() {
  float* v0{new float[]{
    1, 3,
    2, 4}};
  float* v1{new float[]{
    2,0}};
  float* correct{new float[]{
    2,
    4}};
  float* result = new float[2];
  Matrix* list = new Matrix[]{{2,2,v0}, {1,2,v1}, {2,1,result}, {2,1,correct}};
  mulMatABT(list[0], list[1], list[2]);
  return list;
} 

Matrix* getMulTest1ABT() {
  float* v0{new float[]{ 
    0, 2, 3, 
    0 ,0, 0}};
  float* v1{new float[]{ 
    1, 0, 2,
    2 ,3, 1}};
  float* correct = {new float[]{
    6, 9,
    0, 0,}};
  float* result = new float[4]; 
  Matrix * list = new Matrix[]{{2,3,v0}, {2,3,v1}, {2,2,result}, {2,2,correct}};
  mulMatABT(list[0], list[1], list[2]);
  return list;
}

void delete_test_data(Matrix list[]) {
  for(int i = 0; i < 4; i++) { delete list[i].v; }
  delete[] list;
}

Matrix* getMulTest0ATB() {
  float* v0{new float[]{
    1, 2,
    3, 4}};
  float* v1{new float[]{
    2,
    0}};
  float* correct{new float[]{
    2,
    4}};
  float* result = new float[2];
  Matrix* list = new Matrix[]{{2,2,v0}, {2,1,v1}, {2,1,result}, {2,1,correct}};
  mulMatATB(list[0], list[1], list[2]);
  return list;
} 

Matrix* getMulTest1ATB() {
  float* v0{new float[]{ 
    1, 0, 2, 
    2 ,3, 1, 
    0, 0, 0}};
  float* v1{new float[]{  
    0, 2, 3, 
    0 ,0, 1, 
    2, 1, 0}};
  float* correct = {new float[]{
    0, 2, 5, 
    0, 0, 3, 
    0, 4, 7}};
  float* result = new float[9];
  Matrix* list = new Matrix[]{{3,3,v0}, {3,3,v1}, {3,3,result}, {3,3,correct}};
  mulMatATB(list[0], list[1], list[2]);
  return list;
}

Matrix* getAddTest() {
  float* v0{new float[]{ 
    0, 2, 3, 
    0 ,0, 0,
    1, 2, 4}};
  float* v1{new float[]{ 
    1, 2, 2,
    0 ,3, 4, 
    2, 1, 5}};
  float* correct = {new float[]{
    1, 4, 5,
    0, 3, 4,
    3, 3, 9}};
  float* result = new float[9];

  Matrix * list = new Matrix[]{{3,3,v0}, {3,3,v1}, {3,3,result}, {3,3,correct}};
  addMat(list[0], list[1], list[2]);
  return list;
}

void testMatOperation(Matrix* list, std::string text) {
  for(int x = 0; x < list[3].ht; x++) {
    for(int y = 0; y < list[3].wt; y++) {
      if(list[2][x][y] != list[3][x][y]) {

        std::string error_message = text 
          +" with indices " + std::to_string(x) +" "+ std::to_string(y) 
          +"; got: "+ std::to_string(list[2][x][y]) +" expected: "+ std::to_string(list[3][x][y]); 
        TEST(false, error_message);
        return;
      }
    }
  }

  delete_test_data(list);
  std::cout << text << " complete\n";
}

int main(void) {
  testMatOperation(getMulTest0(), "mulMat test 0");
  testMatOperation(getMulTest1(), "mulMat test 1");
  testMatOperation(getMulTest2(), "mulMat test 2");
  testMatOperation(getAddTest(), "addMat test 0");
  testMatOperation(getMulTest0ABT(), "mulTestABT test 0");
  testMatOperation(getMulTest1ABT(), "mulTestABT test 1");
  testMatOperation(getMulTest0ATB(), "mulTestATB test 0");
  testMatOperation(getMulTest1ATB(), "mulTestATB test 1");
}















