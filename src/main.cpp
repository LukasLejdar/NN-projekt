#include <cstdio>
#include <iostream>
#include "network/net.hpp"
#include "../src/network/math.hpp"


int main() {
  int in_shapes[] = {5,4,3,2,1};
  Net net(in_shapes, 5);
  testMulMat();

  return 0;
}
