#include <assert.h>
#include <cmath>
#include <iostream>
#include <cstdio>
#include <random>
#include <omp.h>
#include "math.hpp"

void printMat(Matrix& mat) {
  for(size_t x = 0; x < mat.ht; x++) {
    for(size_t y = 0; y < mat.wt; y++) {
      printf("%6.4lf ", mat[x][y]);
    }
    std::cout << "\n";
  }
}

static std::string grayscale=" .:-=+*#%@";
static int GRAYSCALE_LENGTH = 10;

void drawMat(Matrix &mat) {
  for(size_t x = 0; x < mat.ht; x++) {
    for(size_t y = 0; y < mat.wt; y++) {
      size_t br = mat[x][y]*(GRAYSCALE_LENGTH-1);
      printf("%c", grayscale.at(br));
    }
    std::cout << "\n";
  }

}

void randomizeMat(Matrix& mat) {
  std::default_random_engine generator;
  std::normal_distribution<float> distribution(0, 1.0 / mat.ht);

  for(size_t i = 0; i < mat.wt*mat.ht; i++) {
    mat.v[i] = distribution(generator);
  }
}


Matrix addMat(Matrix& m1, Matrix& m2) {
  float* v = new float[m1.wt*m1.ht];
  Matrix result = {m1.ht, m1.wt, v};
  addMat(m1, m2, result);
  return result;
}

void addMat(Matrix& m1, Matrix& m2, Matrix& result) {
  assert(m1.ht == m2.ht && m2.ht == result.ht && m1.wt == m2.wt && m2.wt == result.wt);

  float *m1i = m1.v, *m2i = m2.v, *resi = result.v;
  for(; m1i < m1.v + m1.wt*m1.ht; m1i++, m2i++, resi++) {
    (*resi) = (*m1i) + (*m2i);
  }
}

