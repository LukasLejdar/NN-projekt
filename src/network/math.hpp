#include <algorithm>
#include <assert.h>
#include <iostream>
#include <new>
#include <stdio.h>
#include <tuple>
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
    size_t ht;
    size_t wt;
    float* v;

    Matrix(): ht(0), wt(0), v(nullptr) {}

    Matrix(const Matrix &other): ht(other.ht), wt(other.wt){
      //std::cout << "copy Matrix " << ht << " - " << wt << "\n"; 
      v = new float[ht*wt];
      std::copy(other.v, other.v+ht*wt, v);
    }

    Matrix(size_t ht, size_t wt, float* v_): ht(ht), wt(wt) {
      v = new float[ht*wt];
      std::copy(v_, v_+ht*wt, v);
    }

    Matrix(size_t ht, size_t wt): ht(ht), wt(wt) {
      //std::cout << "empty initializer " << ht << " - " << wt << "\n";
      v = new float[ht*wt];
      std::fill(v, v+ht*wt, 0);
    }

    ~Matrix() {
      //std::cout << "delete mat " << ht << " - " << wt << "\n";
      delete [] v;
    }

    void faltten() {
      ht = ht*wt;
      wt = 1;
    }

    void swap(Matrix& other) {
      std::swap(other.ht, ht);
      std::swap(other.wt, wt);
      std::swap(other.v, v);
    }

    float* operator[](int p) { return &(v[p*wt]); }
    Matrix& operator*(float scaler) {
      for(size_t i = 0; i < ht*wt; i++) v[i] *= scaler;
      return *this;
    }

    Matrix& operator=(const Matrix& other) {
      //std::cout << "const assignment " << other.ht << " - " << other.wt << "\n";
      Matrix temp(other);
      swap(temp);
      return *this;
    }

    Matrix& operator=(Matrix& other) {
      //std::cout << "non const assignment " << ht << " - " << wt << "\n"; 
      swap(other);
      return *this;
    }
};

void printMat(Matrix& m);
void drawMat(Matrix& m, float sensitivity);
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

/// result = left*right + result
template<size_t tileSize, int scaler=1, bool zero=true>
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
          result.v[i*wt+j] += scaler*(left.v[i*in+k]*right.v[k*wt+j]);
        }
      }
    }
  }  
}

///matrixVector multiplication
template<size_t tileSize>
inline void mulMatAvT(Matrix& left, Matrix& right, Matrix& result) {
  assert(right.ht == 1 || right.wt == 1);
  std::swap(right.ht, right.wt);
  matMul<tileSize>(left, right, result);
  std::swap(right.ht, right.wt);
}


/// result = left^T*right + result
template<size_t tileSize, int scaler=1, bool zero=true>
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
          result.v[i*wt+j] += scaler*(left.v[k*in+i]*right.v[k*wt+j]);
        }
      }
    }
  }
}

template<size_t tileSize, int scaler=1>
inline void addMat(Matrix& m, Matrix& result) {
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

#endif

