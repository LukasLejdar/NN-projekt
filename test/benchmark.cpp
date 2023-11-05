#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cstdio>
#include <iostream>
#include <chrono>
#include <string>
#include "../src/network/math.hpp"

class Timer {
  public:
    Timer(std::string message) {
      const int length = message.length(); 
      this->message = new char[length + 1]; 
      strcpy(this->message, message.c_str());

      startTimePoint = std::chrono::high_resolution_clock::now();
    }

    ~Timer() {
      stop();
    }

    void stop() {
      auto endTimePoint = std::chrono::high_resolution_clock::now();
      auto start = std::chrono::time_point_cast<std::chrono::microseconds>(startTimePoint).time_since_epoch();
      auto end = std::chrono::time_point_cast<std::chrono::microseconds>(endTimePoint).time_since_epoch();

      auto duration = end - start;
      double ms = duration.count() * 0.001;
      printf(message, ms);
    }

  private:
    char* message;
    std::chrono::time_point<std::chrono::high_resolution_clock> startTimePoint;
};

Matrix* getMulTest1() {
  size_t size = 1024; 
  float* v0 = new float[size*size];
  float* v1 = new float[size*size];
  float* result = new float[size*size];
  Matrix* list = new Matrix[]{
    {size, size, v0}, 
    {size, size, v1}, 
    {size, size, result}};
  randomizeMat(list[0]);
  randomizeMat(list[1]);
  return list;
}

Matrix* getMulTest2() {
  size_t wt = 2048; 
  size_t ht = 512; 

  float* v0 = new float[ht*wt];
  float* v1 = new float[ht*wt];
  float* result = new float[ht*wt];
  Matrix* list = new Matrix[]{
    {ht, wt, v0}, 
    {wt, ht, v1}, 
    {wt, ht, result}};
  randomizeMat(list[0]);
  randomizeMat(list[1]);
  return list;
}

void benchMatMul() {
  Matrix* list = getMulTest1();
  Timer timer = Timer("bench matMul 8 time in ms: %0.9f \n");
  addMulMat<1024,1024,1024,8>(list[0], list[1], list[2]);
}

void benchMatMul1() {
  Matrix* list = getMulTest1();
  Timer timer = Timer("bench matMul 16 time in ms: %0.9f \n");
  addMulMat<1024,1024,1024,16>(list[0], list[1], list[2]);
}

void benchMatMul2() {
  Matrix* list = getMulTest1();
  Timer timer = Timer("bench matMul 32 time in ms: %0.9f \n");
  addMulMat<1024,1024,1024,32>(list[0], list[1], list[2]);
}

void benchMatMul3() {
  Matrix* list = getMulTest1();
  Timer timer = Timer("bench matMul 64 time in ms: %0.9f \n");
  addMulMat<1024,1024,1024,64>(list[0], list[1], list[2]);
}

void benchMatMul4() {
  Matrix* list = getMulTest1();
  Timer timer = Timer("bench matMul 128 time in ms: %0.9f \n");
  addMulMat<1024,1024,1024,128>(list[0], list[1], list[2]);
}

void benchMatMulABT() {
    Matrix* list = getMulTest1();
    Timer timer = Timer("bench matMulABT time in ms: %0.9f \n");
    transpose<1024,1024,8>(list[2], list[1]);
    addMulMat<1024,1024,1024,8>(list[0], list[1], list[2]);
}

void benchMatMulATB() {
  Matrix* list = getMulTest1();
  Timer timer = Timer("bench matMulATB time in ms: %0.9f \n");
  addMulMatATB<1024,1024,1024,8>(list[0], list[1], list[2]);
}

void benchTranspose0() {
  Matrix* list = getMulTest1();
  Timer timer = Timer("bench tranpose square matrix time in ms: %0.9f \n");
  transpose<1024,1024,8>(list[0], list[2]);
}

void benchTranspose1() {
  Matrix* list = getMulTest2();
  Timer timer = Timer("bench tranpose rectangular matrix wider time in ms: %0.9f \n");
  transpose<512,2048,8>(list[0], list[2]);
}

void benchTranspose2() {
  Matrix* list = getMulTest2();
  Timer timer = Timer("bench tranpose rectangular matrix taller time in ms: %0.9f \n");
  transpose<2048,512,8>(list[1], list[0]);
}

int main(void) {
  std::cout << "find optimal tailing\n";
  benchMatMul();
  benchMatMul1();
  benchMatMul2();
  benchMatMul3();
  benchMatMul4();

  std::cout << "\n";
  benchTranspose0();
  benchTranspose1();
  benchTranspose2();
  benchMatMulABT();
  benchMatMulATB();
}
