#include <cstddef>
#include <iostream>
#include <math.h>
#include <assert.h>
#include <stdio.h>
#ifndef MATH_H
#define MATH_H

/// Matrix stores its values in a single array v
/// m[x][y] = v[x*width+y]
/// y ----- width
/// x 1 2 3 
/// | 4 5 6 => 1 2 3 4 5 6 7 8 9 
/// | 7 8 9
/// height
class Matrix {
  public:
    size_t ht; //height
    size_t wt; //width
    float* v;

    float* operator[](int p) { return &(v[p*wt]); }
};

void printMat(Matrix& m);
void drawMat(Matrix& m);
void randomizeMat(Matrix& mat);

/// result = left*right + result
template<size_t ht, size_t in, size_t wt, size_t tileSize>
inline void addMulMat(Matrix& left, Matrix& right, Matrix& result) {
  assert(left.wt == right.ht && right.ht == in);
  assert(result.ht == left.ht && left.ht == ht && result.wt == right.wt && right.wt == wt);

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

template<size_t ht, size_t wt, size_t tileSize>
inline void transpose(Matrix& a, Matrix& result) {
  assert(a.wt == wt && a.ht == ht);
  assert(result.ht == wt && result.wt == ht);

  for (size_t htTile = 0; htTile < ht; htTile += tileSize) {

    for(size_t j = 0; j < wt; j++) {
      size_t htTileEnd = std::min(ht, htTile + tileSize);

      for(size_t i = htTile; i < htTileEnd; i++) {
        result.v[j*ht+i] = a.v[i*wt+j];
      }
    }
  }
}

/// result = left^T*right + result
template<size_t wt2, size_t in, size_t wt, size_t tileSize>
inline void addMulMatATB(Matrix& left, Matrix& right, Matrix& result) {
  assert(left.ht == right.ht && right.ht == in);
  assert(result.ht == left.wt && left.wt == wt2 && result.wt == right.wt && right.wt == wt);

  for (size_t wt2Tile = 0; wt2Tile < wt2; wt2Tile += tileSize) {
    for(size_t k = 0; k < in; k++) {
      size_t wt2TileEnd = std::min(wt2, wt2Tile + tileSize);

      for(size_t i = wt2Tile; i < wt2TileEnd; i++) {
        for(size_t j = 0; j < wt; j++) {
          result.v[i*wt+j] += left.v[k*in+i]*right.v[k*in+j];
        }
      }
    }
  }
}

template<size_t ht, size_t wt, size_t tileSize>
inline void addMat(Matrix& m, Matrix& result) {
  assert(m.ht == ht && m.wt == wt);
  assert(result.ht == ht && result.wt == wt);

  for (size_t htTile = 0; htTile < ht; htTile += tileSize) {
    for(size_t j = 0; j < wt; j++) {
      size_t htTileEnd = std::min(ht, htTile + tileSize);

      for(size_t i = htTile; i < htTileEnd; i++) {
        result.v[i*wt+j] += m.v[i*wt+j];
      }
    }
  }
}

#endif

