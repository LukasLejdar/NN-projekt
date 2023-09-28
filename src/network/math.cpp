#include <assert.h>
#include <cstdio>
#include "math.hpp"

void mulMat(Matrix m1, Matrix m2, Matrix result) {
  assert(m1.wt == m2.ht);
  //TODO: deallocate result v
  
  float v[m1.ht*m2.wt];
  result.ht = m1.ht;
  result.wt = m2.wt;
  result.v = v;

  for(int y = 0; y < m1.ht; y++) {
    for(int x = 0; x < m2.wt; x++) {
      float sum = 0;
      for(int i = 0; i < m1.wt; i++) { sum+=m1.v[y*m1.wt+i]*m2.v[i*m2.wt+x]; }
      result.v[y*m2.wt+x]=sum;
    }
  }
}

void addMat(Matrix m1, Matrix m2, Matrix result) {
  assert( m1.ht == m2.ht && m1.wt == m2.wt );

  for(int x = 0; x < m1.wt; x++) {
    for(int y = 0; y < m1.ht; y++) {
      result.v[y*m1.wt+x] = m1.v[y*m1.wt+x] + m2.v[y*m1.wt+x];
    }
  }
}

