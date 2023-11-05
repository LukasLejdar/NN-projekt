#include <thread>
#include <chrono>
#include <omp.h>
#include <iostream>
#include <vector>
#include "../src/network/math.hpp"

//void spawnThreads(int n, Matrix* input) {
//  std::vector<std::thread> threads(n);
//  for (int i = 0; i < n; i++) {
//    threads[i] = std::thread(addMat<3,3,8>, input[i], input[n+1]);
//  }
//
//  for (auto& th : threads) {
//    th.join();
//    printMat(input[n+1]);
//  }
//}

void omp() {
  float* v0 = {new float[]{ 
    1, 0, 0, 
    0 ,0, 0, 
    0, 0, 0}};
  float* v1 = {new float[]{ 
    0, 2, 0, 
    0 ,0, 0, 
    0, 0, 0}};
  float* v2 = {new float[]{ 
    0, 0, 3, 
    0 ,0, 0, 
    0, 0, 0}};
  float* v3 = {new float[]{ 
    0, 0, 0, 
    4 ,0, 0, 
    0, 0, 0}};
  Matrix matrices[] = {{3,3,v0}, {3,3,v1}, {3,3,v2}, {3,3,v3}};

  float* r = {new float[]{ 
    0, 0, 0, 
    0 ,0, 0, 
    0, 0, 0}};
  Matrix result = {3,3,r};

  #pragma omp parallel for
  for(int i = 0; i < 4; i++) {
    std::this_thread::sleep_for(std::chrono::microseconds(i));
    addMat<3,3,8>(matrices[i], result);
    std::cout << omp_get_thread_num() << "\n";
  }
  printMat(result);

}

int main() {
  omp();
}
