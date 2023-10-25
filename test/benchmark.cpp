#include <assert.h>
#include <bits/chrono.h>
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

void benchMatMul() {
    

}

int main(void) {
  benchMatMul();
}
