#include <assert.h>
#include <cstdio>
#include <iostream>
#include "../src/network/math.hpp"

#include <chrono>

class Timer {
  public:
    Timer() {
      startTimePoint = std::chrono::high_resolution_clock::now();
    }
  private:
    std::chrono::time_point<std::chrono::high_resolution_clock> startTimePoint;
};

void benchMatMul() {

}

int main(void) {
  benchMatMul();
}
