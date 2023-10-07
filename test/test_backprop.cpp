#include <assert.h>
#include <cstdio>
#include <iostream>
#include "../src/network/math.hpp"
#include "../src/network/net.hpp"

void test_back_prop() {
  Dense layers[] = {
    DENSE(3, 3, RELU),
    DENSE(3, 2, SOFTMAX),
  };

  Net net(layers, 2);

  float v[] = {0, 0, 1};
  Matrix inp = {3, 1, v};
}

int main(void) {
  test_back_prop();
  return 0;
}

