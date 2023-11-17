#ifndef TEST_H
#define TEST_H

#include "../src/network/math.hpp"
#include <iostream>

#define TEST(x, error_message) { if (!(x)) std::cout << __FUNCTION__ << " failed on line " << __LINE__  << " executing " << error_message << std::endl; }

inline void testMatOperation(Matrix* list, std::string text, float epsilon=0) {
  for(size_t x = 0; x < list[1].ht; x++) {
    for(size_t y = 0; y < list[1].wt; y++) {
      if(std::abs(list[0][x][y] - list[1][x][y]) > epsilon) {
        std::cout << "\ngot: \n";
        printMat(list[0]);
        std::cout << "expected: \n";
        printMat(list[1]);

        std::string error_message = text + " with indices " + std::to_string(x) +" "+ std::to_string(y);
        TEST(false, error_message);
        return;
      }
    }
  }
  
  std::cout << text << " complete\n";
}

#endif
