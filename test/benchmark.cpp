#include <assert.h>
#include <cstdio>
#include <iostream>
#include <chrono>
#include <string>
#include "../src/network/math.hpp"

class Timer {
  public:
    Timer(char* message) {
      this->message = message;
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

  float* v0 = new float[500*500];
  float* v1 = new float[500*500];
  float* result = new float[500*500];
  Matrix* list = new Matrix[]{{500,500,v0}, {500,500,v1}, {500,500,result}};
  randomizeMat(list[0]);
  randomizeMat(list[1]);
  return list;
}

void benchMatMul() {
    Timer timer = Timer("time in ms: %0.9f \n");
    Matrix* list = getMulTest1();
    for(int i = 0; i < 10; i++) {
      mulMat(list[0], list[1], list[2]);
    }
}

int main(void) {
  benchMatMul();
}
