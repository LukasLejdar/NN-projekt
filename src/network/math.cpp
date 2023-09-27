#include <assert.h>
#include <cstdio>
#include "math.hpp"

float* mulMat(float m1[], int h1, int w1, float m2[], int h2, int w2, float result[]) {
  assert(w1==h2);

  for(int y = 0; y < h1; y++) {
    for(int x = 0; x < w2; x++) {
      float sum = 0;
      for(int i = 0; i < w1; i++) { sum+=m1[y*w1+i]*m2[i*w2+x]; }
      result[y*w2+x]=sum;
    }
  }

  return result;
}

