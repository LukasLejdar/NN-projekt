#ifndef MATH_H
#define MATH_H

// mat[y*width+x]
// x -----
// y 1 2 3 
// | 4 5 6 => 1 2 3 4 5 6 7 8 9 
// | 7 8 9
//
// return mat[h1*w2]
float* mulMat(float m1[], int h1, int w1, float m2[], int h2, int w2, float res[]);
void testMulMat();

#endif
