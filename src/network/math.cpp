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

void printMat(Matrix& mat, char separator) {
  for(size_t x = 0; x < mat.ht; x++) {
    for(size_t y = 0; y < mat.wt; y++) {
      printf("%6.14lf%c ", mat[x][y], separator);
    }
    std::cout << "\n";
  }
}

static int GRAYSCALE_LENGTH = 11;
static char* grayscale=(char*) " .:-=+*#%@@";

void drawMat(Matrix &mat, float sensitivity) {
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

std::tuple<float, float> getVarAndExp(Matrix &m) {
    float mean = 0; float varience = 0;
    size_t size = m.wt*m.ht;
    for(size_t l = 0; l < size; l++) mean += m.v[l];
    mean /= size;
    for(size_t l = 0; l < size; l++) varience += pow(m.v[l] - mean, 2);
    varience /= size;
    
    return std::tuple<float, float>(mean, varience);
}

void randomizeMat(Matrix& mat) {
  std::random_device rd;
  std::mt19937 generator(rd());
  std::normal_distribution<float> distribution(0, sqrt(2.0 / (mat.wt + mat.ht)));

  for(size_t i = 0; i < mat.wt*mat.ht; i++) {
    mat.v[i] = distribution(generator);
    //mat.v[i] = rand() % 5 - 2;
  }
}

void zeroMat(Matrix &mat) {
  std::fill(mat.v, mat.v+mat.ht*mat.wt, 0);
}

void copyMatricesOfSameSize(Matrix &from, Matrix &to) {
  assert(from.ht*from.wt == to.ht*to.wt);
  std::copy(from.v, from.v+from.ht*from.wt, to.v);
}
