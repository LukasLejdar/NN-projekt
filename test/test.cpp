#include <assert.h>
#include <cstdio>
#include <iostream>
#include "../src/network/math.hpp"

#define IS_TRUE(x) { if (!(x)) std::cout << __FUNCTION__ << " failed on line " << __LINE__ << std::endl; }


void testMulMat0() {
  float m1[] = {
    1, 3,
    2, 4};
  float m2[] = { 
    2, 
    0};
  float res[9];

  mulMat(m1, 2, 2, m2, 2, 1, res);
  IS_TRUE(res[0]==2 && res[1]==4);

  printf("test 1 complete\n");
}

void testMulMat1() {
  float m1[]= { 
    1, 2, 0, 
    0 ,3, 0, 
    2, 1, 0};
  float m2[] = { 
    0, 2, 3, 
    0 ,0, 1, 
    2, 1, 0};
  float res[9];

  mulMat(m1, 3, 3, m2, 3, 3, res);

  IS_TRUE(res[0]==0 && res[1]==2 && res[2]==5);
  IS_TRUE(res[3]==0 && res[4]==0 && res[5]==3);
  IS_TRUE(res[6]==0 && res[7]==4 && res[8]==7);

  printf("test 2 complete\n");
}

void testMulMat2() {
  float m1[]= { 
    0, 2, 3, 
    0 ,0, 0};
  float m2[] = { 
    1, 2, 
    0 ,3, 
    2, 1};
  float res[9];

  mulMat(m1, 2, 3, m2, 3, 2, res);
  IS_TRUE(res[0]==6 && res[1]==9);
  IS_TRUE(res[2]==0 && res[3]==0);

  printf("test 3 complete\n");
}

int main(void) {
  testMulMat0();
  testMulMat1();
  testMulMat2();
}















