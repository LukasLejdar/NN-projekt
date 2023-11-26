
#include <chrono>
#include <string>
#include "../src/network/math.hpp"

#ifndef BENCHMARK_H
#define BENCHMARK_H
  
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


#endif
