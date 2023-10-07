#include <assert.h>
#include <cmath>
#include <iostream>
#include <cstdio>
#include <random>
#include "math.hpp"

void printMat(Matrix& mat) {
  for(int x = 0; x < mat.ht; x++) {
    for(int y = 0; y < mat.wt; y++) {
      printf("%6.4lf ", mat[x][y]);
    }
    std::cout << "\n";
  }
}

static std::string grayscale=" .:-=+*#%@";
static int GRAYSCALE_LENGTH = 10;

void drawMat(Matrix &mat) {
  for(int x = 0; x < mat.ht; x++) {
    for(int y = 0; y < mat.wt; y++) {
      int br = mat[x][y]*(GRAYSCALE_LENGTH-1);
      printf("%c", grayscale.at(br));
    }
    std::cout << "\n";
  }

}

void randomizeMat(Matrix& mat) {
  std::default_random_engine generator;
  std::normal_distribution<float> distribution(0, 1.0 / mat.ht);

  for(int i = 0; i < mat.wt*mat.ht; i++) {
    mat.v[i] = distribution(generator);
  }
}


Matrix& Matrix::operator*(float f) {
  for(int i = 0; i < ht*wt; i++) {
    v[i] = v[i]*f;
  }
  return *this;
}

// Multiplication -----------------------------------

Matrix Matrix::operator*(Matrix& m) {
  return mulMat(*this, m);
}

Matrix mulMat(Matrix& m1, Matrix& m2) {
  float* v = new float[m1.wt*m2.ht];
  Matrix result = {m1.ht, m2.wt, v};
  mulMat(m1, m2, result);
  return result;
}

void mulMat(Matrix& m1, Matrix& m2, Matrix& result) {
  assert(m1.wt == m2.ht);
  assert(result.ht == m1.ht && result.wt == m2.wt);

  float* resi = result.v;
  for(float* m1i = m1.v; m1i < m1.v + m1.ht*m1.wt; m1i += m1.wt) {
    for(float* m2i = m2.v; m2i < m2.v + m2.wt; m2i++) {
      (*resi) = (*m1i)*(*m2i);
      for(float *_m1i = m1i+1, *_m2i=m2i+m2.wt; _m1i < m1i + m1.wt; _m1i++, _m2i+=m2.wt) {
        (*resi) += (*_m1i)*(*_m2i);
      }
      resi++;
    }
  }
}

void mulMatABT(Matrix& m1, Matrix& m2, Matrix& result) {
  assert(m1.wt == m2.wt);
  assert(result.ht == m1.ht && result.wt == m2.ht);

  float* resi = result.v;
  for(float* m1i = m1.v; m1i < m1.v + m1.ht*m1.wt; m1i += m1.wt) {
    for(float* m2i = m2.v; m2i < m2.v + m2.ht*m2.wt; m2i += m2.wt) {
      (*resi) = (*m1i)*(*m2i);
      for(float *_m1i = m1i+1, *_m2i=m2i+1; _m1i < m1i + m1.wt; _m1i++, _m2i++) {
        (*resi) += (*_m1i)*(*_m2i);
      }
      resi++;
    }
  }
}

void mulMatATB(Matrix& m1, Matrix& m2, Matrix& result) {
  assert(m1.ht == m2.ht);
  assert(result.ht == m1.wt && result.wt == m2.wt);

  float* resi = result.v;
  for(float* m1i = m1.v; m1i < m1.v + m1.wt; m1i ++) {
    for(float* m2i = m2.v; m2i < m2.v + m2.wt; m2i++) {
      (*resi) = (*m1i)*(*m2i);
      for(float *_m1i = m1i+m1.wt, *_m2i=m2i+m2.wt; _m1i < m1i + m1.wt*m1.ht; _m1i += m1.wt, _m2i += m2.wt) {
        (*resi) += (*_m1i)*(*_m2i);
      }
      resi++;
    }
  }
}

// Addition ------------------------------------------

Matrix Matrix::operator+(Matrix& m) {
  return addMat(*this, m);
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

