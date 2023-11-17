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

    inline void stop() {
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
  Matrix* list = new Matrix[]{
    {size, size}, 
    {size, size}, 
    {size, size}};
  randomizeMat(list[0]);
  randomizeMat(list[1]);
  return list;
}

Matrix* getMulTest2() {
  size_t wt = 2048; 
  size_t ht = 512; 
  Matrix* list = new Matrix[]{
    {ht, wt}, 
    {wt, ht}, 
    {wt, ht}};
  randomizeMat(list[0]);
  randomizeMat(list[1]);
  return list;
}

void benchMatMul() {
  Matrix* list = getMulTest1();
  Timer timer = Timer("bench matMul 8: %0.3f ms \n");
  matMul<8>(list[0], list[1], list[2]);

}

void benchMatMul1() {
  Matrix* list = getMulTest1();
  Timer timer = Timer("bench matMul 16: %0.3f ms \n");
  matMul<16>(list[0], list[1], list[2]);
}

void benchMatMul2() {
  Matrix* list = getMulTest1();
  Timer timer = Timer("bench matMul 32: %0.3f ms \n");
  matMul<32>(list[0], list[1], list[2]);
}

void benchMatMul3() {
  Matrix* list = getMulTest1();
  Timer timer = Timer("bench matMul 64: %0.3f ms \n");
  matMul<64>(list[0], list[1], list[2]);
}

void benchMatMul4() {
  Matrix* list = getMulTest1();
  Timer timer = Timer("bench matMul 128: %0.3f ms \n");
  matMul<128>(list[0], list[1], list[2]);
}

void benchMatMulScalerm1() {
  Matrix* list = getMulTest1();
  Timer timer = Timer("bench matMul with scaler -1: %0.3f ms \n");
  matMul<8, -1>(list[0], list[1], list[2]);
}

void benchMatMulScaler1() {
  Matrix* list = getMulTest1();
  Timer timer = Timer("bench matMul with scaler 1: %0.3f ms \n");
  matMul<8, 1>(list[0], list[1], list[2]);
}

void benchMatMulScaler6() {
  Matrix* list = getMulTest1();
  Timer timer = Timer("bench matMul with scaler 6: %0.3f ms \n");
  matMul<8, 6>(list[0], list[1], list[2]);
}

void benchMatMulABT() {
  Matrix* list = getMulTest1();
  Timer timer = Timer("bench matMulABT: %0.3f ms \n");
  transpose<8>(list[2], list[1]);
  matMul<8>(list[0], list[1], list[2]);
}

void benchMatMulATB() {
  Matrix* list = getMulTest1();
  Timer timer = Timer("bench matMulATB: %0.3f ms \n");
  matMulATB<8>(list[0], list[1], list[2]);
}

void benchTranspose0() {
  Matrix* list = getMulTest1();
  Timer timer = Timer("bench transpose square matrix: %0.3f ms \n");
  transpose<8>(list[0], list[2]);
}

void benchTranspose1() {
  Matrix* list = getMulTest2();
  Timer timer = Timer("bench transpose rectangular matrix wider: %0.3f ms \n");
  transpose<8>(list[0], list[2]);
}

void benchTranspose2() {
  Matrix* list = getMulTest2();
  Timer timer = Timer("bench transpose rectangular matrix taller: %0.3f ms \n");
  transpose<8>(list[1], list[0]);
}

void benchAddMat() {
  Matrix* list = getMulTest1();
  Timer timer = Timer("bench add matricies: %0.3f ms \n");
  addMat<8>(list[1], list[0]);
}

void benchZeroMat() {
  Matrix* list = getMulTest1();
  Timer timer = Timer("bench zero mat: %0.3f ms \n");
  zeroMat(list[0]);
  timer.stop();
}

int main(void) {
  std::cout << "find optimal tailing\n";
  benchMatMul();
  benchMatMul1();
  benchMatMul2();
  benchMatMul3();
  benchMatMul4();

  std::cout << "\n";
  benchMatMulScaler1();
  benchMatMulScalerm1();
  benchMatMulScaler6();

  std::cout << "\n";
  benchMatMulABT();
  benchMatMulATB();

  std::cout << "\n";
  benchTranspose0();
  benchTranspose1();
  benchTranspose2();

  std::cout << "\n";
  benchAddMat();
  benchZeroMat();
}
