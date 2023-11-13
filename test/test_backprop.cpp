#include <assert.h>
#include <cstdio>
#include <iostream>
#include "../src/network/math.hpp"
#include "../src/network/net.hpp"

int main(void) {
  MnistReader test_data("mnist/test-images-idx3-ubyte", "mnist/test-labels-idx1-ubyte");
  std::cout << test_data.number_of_entries << "\n";
  std::cout << test_data.last_read.ht << "\n";
  test_data.number_of_entries = 1;


  const size_t LENGTH = 2;
  Dense layers[LENGTH] = {
    {81,4},
    {4,2}
  };

  Net net(layers, LENGTH);
  net.train_epochs(test_data, 2);
 
  return 0;
}

