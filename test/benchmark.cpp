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
  std::size_t size = 1024; 
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

void benchMatMul() {
    Matrix* list = getMulTest1();
    Timer timer = Timer("bench4 time in ms: %0.9f \n");
    mulMat<1024, 1024, 1024, 8>(list[0], list[1], list[2]);
}

int main(void) {
  benchMatMul();
}
