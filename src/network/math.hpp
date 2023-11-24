#include <cassert>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <math.h>
#include <new>
#include <stdio.h>
#include <tuple>
#include "tensor.hpp"

#ifndef MATH_H
#define MATH_H

void printMat(const Matrix& m, char separator='\0');
void drawMat(const Matrix& m, float sensitivity = 1);
void copyMatricesOfSameSize(const Matrix& from, const Matrix& to);
std::tuple<float, float> getVarAndExp(const Matrix& m);

template<size_t tileSize>
inline void transpose(Matrix& a, Matrix& result) {
  size_t ht = a.ht;
  size_t wt = a.wt;

  for (size_t htTile = 0; htTile < ht; htTile += tileSize) {
    for(size_t j = 0; j < wt; j++) {
      size_t htTileEnd = std::min(ht, htTile + tileSize);

      for(size_t i = htTile; i < htTileEnd; i++) {
        result.v[j*ht+i] = a.v[i*wt+j];
      }
    }
  }
}

// Matrix addition -----------------------------------------------------

template<size_t tileSize>
inline void addMat(const Matrix& m, const Matrix& result, float scaler=1) {
  size_t ht = m.ht; assert(m.ht == result.ht);
  size_t wt = m.wt; assert(m.wt == result.wt);

  for (size_t htTile = 0; htTile < ht; htTile += tileSize) {
    for(size_t j = 0; j < wt; j++) {
      size_t htTileEnd = std::min(ht, htTile + tileSize);

      for(size_t i = htTile; i < htTileEnd; i++) {
        result.v[i*wt+j] += scaler*(m.v[i*wt+j]);
      }
    }
  }
}

template<size_t tileSize, int scaler1=1, int scaler2=1>
inline void addMat(const Matrix& m1, const Matrix& m2, const Matrix& result) {
  size_t ht = m1.ht; assert(m1.ht == result.ht && m2.ht == m1.ht);
  size_t wt = m1.wt; assert(m1.wt == result.wt && m2.wt == m1.wt);

  for (size_t htTile = 0; htTile < ht; htTile += tileSize) {
    for(size_t j = 0; j < wt; j++) {
      size_t htTileEnd = std::min(ht, htTile + tileSize);

      for(size_t i = htTile; i < htTileEnd; i++) {
        result.v[i*wt+j] = scaler1*m1.v[i*wt+j] + scaler2*m2.v[i*wt+j];
      }
    }
  }
}

// Matrix multiplication -----------------------------------------------------

template<size_t tileSize, bool zero=true>
inline void matMul(Matrix& left, Matrix& right, Matrix& result) {
  size_t ht = left.ht; assert(result.ht == left.ht);
  size_t in = left.wt; assert(left.wt == right.ht);
  size_t wt = right.wt; assert(result.wt == right.wt);

  if(zero) result.zero();

  for (size_t innerTile = 0; innerTile < in; innerTile += tileSize) {
    for(size_t i = 0; i < ht; i++) {
      size_t innerTileEnd = std::min(in, innerTile + tileSize);
      for(size_t k = innerTile; k < innerTileEnd; k++) {
        for(size_t j = 0; j < wt; j++) {
          result.v[i*wt+j] += left.v[i*in+k]*right.v[k*wt+j];
        }
      }
    }
  }  
}
template<size_t tileSize, bool zero=true>
inline void matMulAv(Matrix& left, Matrix& right, Matrix& result) {
  assert(result.wt == 1 && right.wt == 1);
  size_t ht = left.ht; assert(result.ht == left.ht);
  size_t in = left.wt; assert(left.wt == right.ht);

  if(zero) result.zero();

  for(size_t i = 0; i < ht; i++) {
    for(size_t k = 0; k < in; k++) {
      result.v[i] += left.v[i*in+k]*right.v[k];
    }
  }
}

template<size_t tileSize, bool zero=true>
inline void matMulvvT(Matrix& left, Matrix& right, Matrix& result) {
  assert(left.ht == result.ht && left.wt == 1);
  assert(right.ht == result.wt && right.wt == 1);
  
  for(size_t i = 0; i < left.ht; i++) {
    for(size_t j = 0; j < right.ht; j++) {
      result.v[i*result.wt +j] += left.v[i]*right.v[j];
    }
  }
}


template<size_t tileSize, bool zero=true>
inline void matMulATv(const Matrix& left, const Matrix& right, Matrix& result) {
  assert(result.wt == 1 && result.wt == 1);
  size_t ht = left.wt; assert(left.wt == result.ht);
  size_t in = left.ht; assert(left.ht == right.ht);

  if(zero) result.zero();

  for(size_t i = 0; i < ht; i++) {
    for(size_t k = 0; k < in; k++) {
      result.v[i] += left.v[k*ht+i]*right.v[k];
    }
  }
}

// convolutions -------------------------------------------------------------- 


template<size_t tileSize>
inline void convolveFull(Matrix& kernel, Matrix& input, Matrix& result) {
  assert(input.ht + kernel.ht -1 == result.ht);
  assert(input.wt + kernel.wt -1 == result.wt);

  for(size_t i = 0; i < result.ht; i++) {
    size_t kEnd = fmin(kernel.ht, i+1);
    size_t kBeg = fmax(0, i-input.ht+1);

    for(size_t j = 0; j < result.wt; j++) {
      size_t lEnd = fmin(kernel.wt, j+1);
      size_t lBeg = fmax(0, j-input.wt+1);

      for(size_t k = kBeg; k < kEnd; k++) {
        for(size_t l = lBeg; l < lEnd; l++) {
          result.v[i*result.wt + j] += input.v[(i-k)*input.wt +j -l]* kernel.v[k*kernel.wt+l];
        }
      }
    }
  }
}

template<size_t tileSize>
inline void correlate(Matrix& kernel, Matrix& input, Matrix& result) {
  assert(input.ht - kernel.ht + 1 == result.ht);
  assert(input.wt - kernel.wt + 1 == result.wt);

  for(size_t i = 0; i < input.ht; i++) {
    size_t kEnd = fmin(kernel.ht, i+1);
    size_t kBeg = fmax(0, i-result.ht+1);

    for(size_t j = 0; j < input.wt; j++) {
      size_t lEnd = fmin(kernel.wt, j+1);
      size_t lBeg = fmax(0, j-result.wt+1);

      for(size_t k = kBeg; k < kEnd; k++) {
        for(size_t l = lBeg; l < lEnd; l++) {
          result.v[(i-k)*result.wt+j-l] += input.v[i*input.wt+j]* kernel.v[k*kernel.wt+l];
        }
      }
    }
  }
}

template<size_t tileSize, bool zero=true>
void correlateAv(Tensor<4>& kernel, Tensor<3>& input, Tensor<3>& result) {
  assert(kernel.shape[0] == result.shape[0]);
  assert(kernel.shape[1] == input.shape[0]);

  if(zero) result.zero();

  Matrix inp = input[0];
  Matrix ker = kernel[0][0];
  Matrix res = result[0];

  for(size_t k = 0; k < kernel.shape[1]; k++) {
    for(size_t y = 0; y < kernel.shape[0]; y++) {
      inp.v = &input.v[k*input.ht*input.wt];
      ker.v = &kernel.v[(y*kernel.shape[1] +k)*kernel.ht*kernel.wt];
      res.v = &result.v[y*result.ht*result.wt];
      correlate<8>(ker, inp, res);
    }
  }
}


template<size_t tileSize, bool zero=true>
void correlatevvT(Tensor<3>& kernel, Tensor<3>& input, Tensor<4>& result) {
  assert(kernel.shape[0] == result.shape[0]);
  assert(input.shape[0] == result.shape[1]);

  if(zero) result.zero();

  Matrix inp = input[0];
  Matrix ker = kernel[0];
  Matrix res = result[0][0];
  
  for(size_t y = 0; y < result.shape[0]; y++) {
    for(size_t x = 0; x < result.shape[1]; x++) {
      ker.v = &kernel.v[y*kernel.ht*kernel.wt];
      inp.v = &input.v[x*input.ht*input.wt];
      res.v = &result.v[(y*result.shape[1] +x)*result.ht*result.wt];
      correlate<8>(ker, inp, res);
    }
  }
}

template<size_t tileSize, bool zero=true>
void convolveATv(Tensor<4>& kernel, Tensor<3>& input, Tensor<3>& result) {
  assert(kernel.shape[0] == input.shape[0]);
  assert(kernel.shape[1] == result.shape[0]);

  if(zero) result.zero();

  Matrix inp = input[0];
  Matrix ker = kernel[0][0];
  Matrix res = result[0];

  for(size_t y = 0; y < kernel.shape[1]; y++) {
    for(size_t r = 0; r < kernel.shape[0]; r++) {
      inp.v = &input.v[r*input.ht*input.wt];
      ker.v = &kernel.v[(y*kernel.shape[1] +r)*kernel.ht*kernel.wt];
      res.v = &result.v[y*result.ht*result.wt];
      convolveFull<tileSize>(ker, inp, res);
    }
  }
}



inline void adam(const Matrix& dw, const Matrix& ema, const Matrix& ma, float learning_rate, float decay_rate1, float decay_rate2, size_t t) {
    assert(dw.size == ema.size && dw.size == ma.size);

    for(size_t i = 0; i < dw.size; i++) {
      ma.v[i] =  decay_rate1*ma.v[i] + (1-decay_rate1)*dw.v[i];
      ema.v[i] =  decay_rate2*ema.v[i] + (1-decay_rate2)*pow(dw.v[i], 2);
      ma.v[i] = ma.v[i] / (1 - pow(decay_rate1, t));
      ema.v[i] = ema.v[i] / (1 - pow(decay_rate2, t));

      dw.v[i] = ma.v[i]* -learning_rate/(sqrt(ema.v[i]) + 0.00000001);
    }
}

#endif

