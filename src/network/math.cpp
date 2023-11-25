#include <algorithm>
#include <assert.h>
#include <cmath>
#include <iostream>
#include <cstdio>
#include <random>
#include <omp.h>
#include <math.h>
#include <tuple>
#include "math.hpp"

void printVec(const Vector& vec, char separator) {
  for(size_t i = 0; i < vec.size; i++) printf("%6.4lf%c ", vec[i], separator);
  std::cout << "\n";
}

void printMat(const Matrix& mat, char separator) {
  for(size_t x = 0; x < mat.ht; x++) {
    for(size_t y = 0; y < mat.wt; y++) {
      printf("%6.4lf%c ", mat[x][y], separator);
    }
    std::cout << "\n";
  }
}

static int GRAYSCALE_LENGTH = 11;
static char* grayscale=(char*) " .:-=+*#%@@";

void drawMat(const Matrix &mat, float sensitivity) {
  for(size_t x = 0; x < mat.ht; x++) {
    for(size_t y = 0; y < mat.wt; y++) {
      if(mat[x][y] != mat[x][y]) {
        std::cout << "nan";
        continue;
      }
      float norm = erf(abs(-mat[x][y]*sensitivity));
      int br = norm*(GRAYSCALE_LENGTH-1);
      br = fmax(0, fmin(GRAYSCALE_LENGTH, br));
      std::cout << grayscale[br];
    }
    std::cout << "\n";
  }

}

void copyMatricesOfSameSize(const Matrix &from, const Matrix &to) {
  assert(from.ht*from.wt == to.ht*to.wt);
  std::copy(from.v, from.v+from.ht*from.wt, to.v);
}
