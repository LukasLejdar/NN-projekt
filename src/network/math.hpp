#ifndef MATH_H
#define MATH_H

#define HT 0;
#define WT 1;

// v[y*width+x]
// x ----- width
// y 1 2 3 
// | 4 5 6 => 1 2 3 4 5 6 7 8 9 
// | 7 8 9
// height
struct Matrix {
  int ht; //height
  int wt; //width
  float* v;
};

void mulMat(Matrix m1, Matrix m2, Matrix result);
void addMat(Matrix m1, Matrix m2, Matrix result);

#endif
