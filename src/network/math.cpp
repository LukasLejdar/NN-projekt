#include "math.hpp"
#include <algorithm>
#include <assert.h>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <iostream>
#include <math.h>
#include <omp.h>
#include <random>
#include <tuple>
#include <iostream>
#include <sstream>

#define RESET "\033[0m"

void drawPixel(float pixel) {
  int br = erf(1.2*pixel)*255;
  if (br >= 0) {
    std::cout << "\033[48;2;" << br << ";" << br << ";" << br << "m \033[0m";
  } else {
    std::cout << "\033[48;2;" <<  abs((int) (br*0.8)) << ";" << abs((int) (br*0.8)) << ";" << abs(br) << "m \033[0m";
  }
}

void drawMat(const Matrix &mat) {
  for (size_t x = 0; x < mat.ht; x++) {
    for (size_t y = 0; y < mat.wt; y++) {
      if (mat[x][y] != mat[x][y]) {
        std::cout << "nan";
        continue;
      }

      drawPixel(mat[x][y]);
    }
    std::cout << RESET << "\n";
  }
}

void drawKernels(const Tensor<4> &t) {
  for (size_t y = 0; y < t.shape[1]; y++) {
    for (size_t j = 0; j < t.wt; j++) {
      for (size_t x = 0; x < t.shape[0]; x++) {
        for (size_t i = 0; i < t.ht; i++) {
          if (t[x][y][i][j] != t[x][y][i][j]) {
            std::cout << "nan";
            continue;
          }
          drawPixel(t[x][y][i][j]);
        }
        std::cout << "\033[0m   ";
      }
      std::cout << "\033[0m\n";
    }
    std::cout << "\033[0m\n";
  }
}

void draw3D(const Tensor<3> &t, size_t from, size_t max) {
  for (size_t i = 0; i < t.ht; i++) {
    for (size_t x = from; x < from + std::min<size_t>(t.shape[0], max); x++) {
      for (size_t j = 0; j < t.wt; j++) {
        if (t[x][i][j] != t[x][i][j]) {
          std::cout << "nan";
          continue;
        }

        drawPixel(t[x][i][j]);
      }
      std::cout << "\033[0m   ";
    }
    std::cout << RESET << "\n";
  }
}

void printVec(const Vector &vec, char separator) {
  for (size_t i = 0; i < vec.size; i++)
    printf("%6.4lf%c ", vec[i], separator);
  std::cout << "\n";
}

void printMat(const Matrix &mat, char separator) {
  for (size_t x = 0; x < mat.ht; x++) {
    for (size_t y = 0; y < mat.wt; y++) {
      printf("%6.4lf%c ", mat[x][y], separator);
    }
    std::cout << "\n";
  }
}
