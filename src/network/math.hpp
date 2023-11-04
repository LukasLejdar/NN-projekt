#include <cstddef>
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
    std::size_t ht; //height
    std::size_t wt; //width
    float* v;

    float* operator[](int p) { return &(v[p*wt]); }
    Matrix operator*(Matrix& m);
    Matrix& operator*(float f);
    Matrix operator+(Matrix& m);
    Matrix operator-(Matrix& m);

    //TODO: implement copy operator
};

void printMat(Matrix& m);
void drawMat(Matrix& m);
void randomizeMat(Matrix& mat);

Matrix mulMat(Matrix& m1, Matrix& m2);
void mulMat(Matrix& m1, Matrix& m2, Matrix& result);  // save result to result
void mulMat2(Matrix& m1, Matrix& m2, Matrix& result);  // save result to result

void mulMatABT(Matrix& m1, Matrix& m2, Matrix& result);  // save result to result
void mulMatATB(Matrix& m1, Matrix& m2, Matrix& result);  // save result to result

Matrix addMat(Matrix& m1, Matrix& m2);
void addMat(Matrix& m1, Matrix& m2, Matrix& result); // save result to result

#endif

