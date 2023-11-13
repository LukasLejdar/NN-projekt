#include <algorithm>
#include <list>
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
//

struct Test {
  void execute_on_thread(Matrix* m, Matrix* result, int thread_i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10*thread_i));
    addMat<8>(*m, *result);
    addMat<8>(*m, *result);
    addMat<8>(*m, *result);
    addMat<8>(*m, *result);
  }
};

void omp() {
  const int n_threads = 4;
  size_t wt = 16000;
  size_t ht = 16000;
  Matrix list[n_threads] = {};

  for(size_t i = 0; i < n_threads; i++) {
    list[i] = Matrix(ht, wt);
    randomizeMat(list[i]);
  }

  Matrix result = Matrix(ht, wt);
  Matrix control = Matrix(ht, wt);

  zeroMat(result);
  zeroMat(control);

  std::cout << "begin\n";

  std::thread threads[n_threads];
  for(int i = 0; i < n_threads; i++) {

    threads[i] = std::thread(execute_on_thread, &(list[i]), &result, i);
  }

  for (auto& th : threads) 
    th.join();

  std::cout << "done\n";

  for(int i = 0; i < n_threads; i++) {
    addMat<8>(list[i], control);
    addMat<8>(list[i], control);
    addMat<8>(list[i], control);
    addMat<8>(list[i], control);
  }

  size_t count = 0;
  for(size_t x = 0; x < control.ht; x++) {
    for(size_t y = 0; y < control.wt; y++) {
      if(control[x][y] != result[x][y]) {
        count++;
      }
    }
  }

  std::cout << count << "\n";

}

int main() {
  omp();
}
