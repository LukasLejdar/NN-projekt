#ifndef MNIST_READER 
#define MNIST_READER

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <ios>
#include <iostream>
#include <mutex>
#include <string>
#include <fstream>
#include "network/math.hpp"

class MnistReader {
  public:
    size_t number_of_entries;
    const Tensor<3> images;
    const TensorT<size_t, 1> labels;
    TensorT<size_t, 1> permutation;
    //Shape<2> augmentations[9] = {{0,0}};
    //Shape<2> augmentations[9] = {{0,0}, {1,0}, {1,-1},
    //                             {0,-1}, {-1,-1}, {-1,0},
    //                             {-1,1}, {0,1}, {1,1}};
    Shape<2> augmentations[9] = {{0,0}, {1,0}, {-1,0}};

    MnistReader(std::string images_path, std::string labels_path, Shape<2>, size_t number_of_entries);
    MnistReader(MnistReader& reader, size_t from, size_t to);

    Matrix last_read;
    int index = -1;
    int last_lable;
    bool read_next(bool augment);
    void loop_to_beg();
    void shuffle();

  private:
    std::string images_path;
    std::string labels_path;
};

void augment(Shape<2>& shape, float inp[], Tensor<2>& res);
Tensor<3>& readMnistImagesCsv(std::string file_path, Shape<2> shape, size_t number_of_entries);
TensorT<size_t, 1>& readMnistLablesCsv(std::string file_path, size_t number_of_entries);

#endif
