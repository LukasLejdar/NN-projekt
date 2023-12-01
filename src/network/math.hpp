#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <math.h>
#include <new>
#include <stdio.h>
#include <tuple>
#include "tensor.hpp"
#include <immintrin.h>
#include <iostream>

#ifndef MATH_H
#define MATH_H

void printVec(const Vector& m, char separator='\0');
void printMat(const Matrix& m, char separator='\0');
void drawMat(const Matrix& m);
void drawKernels(const Tensor<4>& t);
void draw3D(const Tensor<3>& t, size_t form = 0, size_t max = 7);

template<size_t dim1, size_t dim2>
void copyToTensorOfSameSize(const Tensor<dim1>& from, const Tensor<dim2>& to) {
  assert(from.size == to.size);
  std::copy(from.v, from.v+from.size, to.v);
}

template<size_t dim>
std::tuple<float, float> getVarAndExp(const Tensor<dim>& t) {
  float mean = 0, varience = 0;

  for(size_t i = 0; i < t.size; i++) 
    mean += t.v[i]; 
  mean /= t.size;

  for(size_t i = 0; i < t.size; i++) 
    varience += pow(t.v[i] - mean, 2);
  varience /= t.size;

  return std::tuple<float, float>(mean, varience);
}

template<size_t dim>
void zero(const Tensor<dim>& t) {
  std::fill(t.v, t.v+t.size, 0);
}

template<size_t dim>
void zero(const TensorT<int, dim>& t) {
  std::fill(t.v, t.v+t.size, 0);
}


template<size_t dim>
void randomize(const Tensor<dim>& t, float mean=0, float variance=1) {
  std::random_device rd;
  std::mt19937 generator(rd());
  std::normal_distribution<float> distribution(mean, std::sqrt(variance));

  for(size_t i = 0; i < t.size; i++) t.v[i] = distribution(generator);
}

template<size_t tileSize>
void transpose(Matrix& a, Matrix& result) {
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

template<size_t dim>
inline void addTens(const Tensor<dim>& inp, const Tensor<dim>& res) {
  assert(inp.size == res.size);
  for(size_t i = 0; i < inp.size; i ++) {
    res.v[i] += inp.v[i];
  }
}

template<size_t dim>
inline void addTens(const TensorT<int, dim>& inp, const TensorT<int, dim>& res) {
  assert(inp.size == res.size);
  for(size_t i = 0; i < inp.size; i ++) {
    res.v[i] += inp.v[i];
  }
}

template<size_t dim>
inline void addTens(const Tensor<dim>& t0, const Tensor<dim>& t1, const Tensor<dim>& result) {
  assert(t0.size == t1.size && t1.size == result.size);
  for(int i = 0; i < result.size; i ++) {
    result.v[i] += t0.v[i] + t1.v[0];
  }
}

// Matrix multiplication -----------------------------------------------------

template<size_t tileSize, bool to_zero=true>
inline void matMul(Matrix& left, Matrix& right, Matrix& result) {
  size_t ht = left.ht; assert(result.ht == left.ht);
  size_t in = left.wt; assert(left.wt == right.ht);
  size_t wt = right.wt; assert(result.wt == right.wt);

  if(to_zero) zero(result);

  for (size_t innerTile = 0; innerTile < in; innerTile += tileSize) {
    size_t innerTileEnd = std::min(in, innerTile + tileSize);

    for(size_t i = 0; i < ht; i++) {
      for(size_t k = innerTile; k < innerTileEnd; k++) {
        for(size_t j = 0; j < wt; j++) {
          result.v[i*wt+j] += left.v[i*in+k]*right.v[k*wt+j];
        }
      }
    }
  }  
}
template<bool to_zero=true>
inline void matMulAv(Matrix& left, Vector& right, Vector& result) {
  size_t ht = left.ht; assert(result.size == left.ht);
  size_t in = left.wt; assert(left.wt == right.size);

  if(to_zero) zero(result);

  for(size_t i = 0; i < ht; i++) {
    for(size_t k = 0; k < in; k++) {
      result.v[i] += left.v[i*in+k]*right.v[k];
    }
  }
}

template<bool zero=true>
inline void matMulvvT(Vector& left, Vector& right, Matrix& result) {
  assert(left.size == result.ht);
  assert(right.size == result.wt);
  
  for(size_t i = 0; i < left.size; i++) {
    for(size_t j = 0; j < right.size; j++) {
      result.v[i*result.wt +j] += left.v[i]*right.v[j];
    }
  }
}


template<bool to_zero=true>
inline void matMulATv(Matrix& left, Vector& right, Vector& result) {
  size_t ht = left.wt; assert(left.wt == result.size);
  size_t in = left.ht; assert(left.ht == right.size);

  if(to_zero) zero(result);

  for(size_t i = 0; i < ht; i++) {
    for(size_t k = 0; k < in; k++) {
      result.v[i] += left.v[k*ht+i]*right.v[k];
    }
  }
}

// convolutions -------------------------------------------------------------- 

inline void convolveFull(Matrix& kernel, Matrix& input, Matrix& result) {
  assert(input.ht + kernel.ht -1 == result.ht);
  assert(input.wt + kernel.wt -1 == result.wt);

  for(size_t i = 0; i < input.ht; i++) {
    for(size_t k = kernel.ht-1; k != SIZE_MAX; k--) {
      for(size_t l = kernel.wt-1; l != SIZE_MAX; l--) {
        for(size_t j = 0; j < input.wt; j++) {
          result.v[(i+k)*result.wt +j+l] += input.v[i*input.wt + j] * kernel.v[k*kernel.wt + l];
        }
      }
    }
  }
}

inline void correlate(Matrix& kernel, Matrix& input, Matrix& result) {
  assert(input.ht - kernel.ht + 1 == result.ht);
  assert(input.wt - kernel.wt + 1 == result.wt);

  for(size_t i = 0; i < result.ht; i++) {
    for(size_t k = 0; k < kernel.ht; k++) {
      for(size_t l = 0; l < kernel.wt; l++) {
        for(size_t j = 0; j < result.wt; j++) {
          result.v[i*result.wt+j] += input.v[(i+k)*input.wt+(j+l)] * kernel.v[k*kernel.wt+l];
        }
      }
    }
  }
}

template<bool to_zero=true>
void correlateAv(Tensor<4>& kernel, Tensor<3>& input, Tensor<3>& result) {
  assert(kernel.shape[0] == result.shape[0]);
  assert(kernel.shape[1] == input.shape[0]);

  if(to_zero) zero(result);

  Matrix inp = input[0];
  Matrix ker = kernel[0][0];
  Matrix res = result[0];

  for(size_t k = 0; k < kernel.shape[1]; k++) {
    for(size_t y = 0; y < kernel.shape[0]; y++) {
      inp.v = &input.v[k*input.ht*input.wt];
      ker.v = &kernel.v[(y*kernel.shape[1] +k)*kernel.ht*kernel.wt];
      res.v = &result.v[y*result.ht*result.wt];
      correlate(ker, inp, res);
    }
  }
}


template<bool to_zero=true>
void correlatevvT(Tensor<3>& kernel, Tensor<3>& input, Tensor<4>& result) {
  assert(kernel.shape[0] == result.shape[0]);
  assert(input.shape[0] == result.shape[1]);

  if(to_zero) zero(result);

  Matrix inp = input[0];
  Matrix ker = kernel[0];
  Matrix res = result[0][0];
  
  for(size_t y = 0; y < result.shape[0]; y++) {
    for(size_t x = 0; x < result.shape[1]; x++) {
      ker.v = &kernel.v[y*kernel.ht*kernel.wt];
      inp.v = &input.v[x*input.ht*input.wt];
      res.v = &result.v[(y*result.shape[1] +x)*result.ht*result.wt];
      correlate(ker, inp, res);
    }
  }
}

template<bool to_zero=true>
void convolveATv(Tensor<4>& kernel, Tensor<3>& input, Tensor<3>& result) {
  assert(kernel.shape[0] == input.shape[0]);
  assert(kernel.shape[1] == result.shape[0]);

  if(to_zero) zero(result);

  Matrix inp = input[0];
  Matrix ker = kernel[0][0];
  Matrix res = result[0];

  for(size_t y = 0; y < kernel.shape[0]; y++) {
    for(size_t x = 0; x < kernel.shape[1]; x++) {
      inp.v = &input.v[y*input.ht*input.wt];
      ker.v = &kernel.v[(y*kernel.shape[1] +x)*kernel.ht*kernel.wt];
      res.v = &result.v[x*result.ht*result.wt];
      convolveFull(ker, inp, res);
    }
  }
}

inline void maxPooling(Tensor<3>& A, Shape<2>& kernel, Tensor<3>& result, TensorT<size_t, 3>& max_locations) {
  assert(A.shape[0] == result.shape[0]);
  assert(result.shape == max_locations.shape);
  assert(result.ht  == std::ceil(A.ht / (float) kernel.ht));
  assert(result.wt  == std::ceil(A.wt / (float) kernel.wt));

  zero(result);

  for(size_t n = 0; n < A.shape[0]; n++) {
    float* Av = A.v + n*A.ht*A.wt;
    float* resv = result.v + n*result.ht*result.wt;
    size_t* maxv = max_locations.v + n*result.ht*result.wt;
    size_t beg = n*A.ht*A.wt;
  
    int y = -1;
    for(size_t i = 0; i < result.ht; i++) {
      for(size_t k = 0; k < kernel.ht && y+1 < (int) A.ht; k++)  {
        y++;
        int x = -1;
        for(size_t j = 0; j < result.wt; j++) {
          for(size_t l = 0; l < kernel.wt && x+1 < (int) A.wt; l++) {
            x++;

            if(Av[y*A.wt + x] > resv[i*result.wt + j]) {
              maxv[i*result.wt + j] = beg + y*A.wt + x;
              resv[i*result.wt + j] = Av[y*A.wt + x];
            }
          }
        }
      }
    }
  }
}

inline void maxPooling_backward(Tensor<3>& dAin, Tensor<3>& dAout, TensorT<size_t, 3>& max_locations) {
  assert(dAin.shape[0] == dAout.shape[0]);
  zero(dAout);
  for(size_t i = 0; i < dAin.size; i++) dAout.v[max_locations.v[i]] = dAin.v[i];
}


#endif

