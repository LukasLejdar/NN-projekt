#include <algorithm>
#include <assert.h>
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

void printMat(Matrix& m, char separator='\0');
void drawMat(Matrix& m, float sensitivity = 1);
void randomizeMat(Matrix& m);
void zeroMat(Matrix& m);
void copyMatricesOfSameSize(Matrix& from, Matrix& to);
std::tuple<float, float> getVarAndExp(Matrix& m);

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

template<size_t tileSize, bool zero=true>
inline void matMul(Matrix& left, Matrix& right, Matrix& result) {
  size_t ht = left.ht; assert(result.ht == left.ht);
  size_t in = left.wt; assert(left.wt == right.ht);
  size_t wt = right.wt; assert(result.wt == right.wt);

  if(zero) zeroMat(result);

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

template<size_t tileSize>
void correlate(Matrix& input, Matrix& kernel, Matrix& result) {
  assert(input.ht - kernel.ht + 1 == result.ht);
  assert(input.wt - kernel.wt + 1 == result.wt);

  zeroMat(result);

  for(size_t i = 0; i < input.ht; i++) {
    for(size_t j = 0; j < input.wt; j++) {
      for(size_t k = fmax(0, i-result.ht+1); k < fmin(kernel.ht, i+1); k++) {
        for(size_t l = fmax(0, j-result.wt+1); l < fmin(kernel.wt, j+1); l++) {
          result.v[(i-k)*result.wt+j-l] += input.v[i*input.wt+j]* kernel.v[k*kernel.wt+l];
        }
      }
    }
  }

}

template<size_t tileSize, bool zero=true>
void convolveFull(Matrix& input, Matrix& kernel, Matrix& result) {
  assert(input.ht + kernel.ht -1 == result.ht);
  assert(input.wt + kernel.wt -1 == result.wt);

  zeroMat(result);

  for(size_t i = 0; i < result.ht; i++) {
    for(size_t j = 0; j < result.wt; j++) {
      for(size_t k = fmax(0, i-input.ht+1); k < fmin(kernel.ht, i+1); k++) {
        for(size_t l = fmax(0, j-input.wt+1); l < fmin(kernel.wt, j+1); l++) {
          result.v[i*result.wt + j] += input.v[(i-k)*input.wt +j -l]* kernel.v[k*kernel.wt+l];
        }
      }
    }
  }
}

///matrixVector multiplication
template<size_t tileSize, bool zero=true>
inline void mulMatAvT(Matrix& left, Matrix& right, Matrix& result) {
  assert(right.ht == 1 || right.wt == 1);
  std::swap(right.ht, right.wt);
  matMul<tileSize, zero>(left, right, result);
  std::swap(right.ht, right.wt);
}


template<size_t tileSize, bool zero=true>
inline void matMulATB(Matrix& left, Matrix& right, Matrix& result) {
  size_t wt2 = left.wt; assert(result.ht == left.wt);
  size_t in = left.ht; assert(left.ht == right.ht);
  size_t wt = right.wt; assert(result.wt == right.wt);

  if(zero) zeroMat(result);

  for (size_t wt2Tile = 0; wt2Tile < wt2; wt2Tile += tileSize) {
    for(size_t k = 0; k < in; k++) {
      size_t wt2TileEnd = std::min(wt2, wt2Tile + tileSize);

      for(size_t i = wt2Tile; i < wt2TileEnd; i++) {
        for(size_t j = 0; j < wt; j++) {
          result.v[i*wt+j] += left.v[k*wt2+i]*right.v[k*wt+j];
        }
      }
    }
  }
}

template<size_t tileSize>
inline void addMat(Matrix& m, Matrix& result, float scaler=1) {
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
inline void addMat(Matrix& m1, Matrix& m2, Matrix& result) {
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

inline void adam(Matrix& dw, Matrix& ema, Matrix& ma, float learning_rate, float decay_rate1, float decay_rate2, size_t t) {
    assert(dw.wt == ema.wt && dw.ht == ema.ht);

    for(size_t i = 0; i < dw.ht*dw.wt; i++) {
      ma.v[i] =  decay_rate1*ma.v[i] + (1-decay_rate1)*dw.v[i];
      ema.v[i] =  decay_rate2*ema.v[i] + (1-decay_rate2)*pow(dw.v[i], 2);
      ma.v[i] = ma.v[i] / (1 - pow(decay_rate1, t));
      ema.v[i] = ema.v[i] / (1 - pow(decay_rate2, t));

      dw.v[i] = ma.v[i]* -learning_rate/(sqrt(ema.v[i]) + 0.00000001);
    }
}
#endif

