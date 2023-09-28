#include <assert.h>
#include <cstdio>
#include <iostream>
#include "../src/network/math.hpp"

#define IS_TRUE(x) { if (!(x)) std::cout << __FUNCTION__ << " failed on line " << __LINE__ << std::endl; }

void testMulMat0() {
  float v1[] = {1, 3,
                2, 4};
  float v2[] = {2,
                0};
  struct Matrix m1 = {2, 2, v1};
  struct Matrix m2 = {2, 1, v2};
  struct Matrix res;

  mulMat(m1, m2, res);
  IS_TRUE(res.v[0]==2 && res.v[1]==4);

  printf("test 1 complete\n");
}

void testMulMat1() {
  float v1[]= { 
    1, 2, 0, 
    0 ,3, 0, 
    2, 1, 0};
  float v2[] = { 
    0, 2, 3, 
    0 ,0, 1, 
    2, 1, 0};
  Matrix m1 = {3, 3, v1};
  Matrix m2 = {3, 3, v2};
  Matrix res;

  mulMat(m1, m2, res);

  IS_TRUE(res.v[0]==0 && res.v[1]==2 && res.v[2]==5);
  IS_TRUE(res.v[3]==0 && res.v[4]==0 && res.v[5]==3);
  IS_TRUE(res.v[6]==0 && res.v[7]==4 && res.v[8]==7);

  printf("test 2 complete\n");
}

void testMulMat2() {
  float v1[]= { 
    0, 2, 3, 
    0 ,0, 0};
  float v2[] = { 
    1, 2, 
    0 ,3, 
    2, 1};
  Matrix m1 = {3, 3, v1};
  Matrix m2 = {3, 3, v2};
  Matrix res;

  mulMat(m1, m2, res);
  IS_TRUE(res.v[0]==6 && res.v[1]==9);
  IS_TRUE(res.v[2]==0 && res.v[3]==0);

  printf("test 3 complete\n");
}

int main(void) {
  testMulMat0();
  testMulMat1();
  testMulMat2();
}















