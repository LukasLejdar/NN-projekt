#ifndef MNIST_READER
#define MNIST_READER

#include "network/math.hpp"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <fstream>
#include <ios>
#include <iostream>
#include <mutex>
#include <string>

#define NAUGMENTATIONS 4

Tensor<3> &readMnistImagesCsv(std::string file_path, Shape<2> shape, size_t number_of_entries);
TensorT<size_t, 1> &readMnistLablesCsv(std::string file_path, size_t number_of_entries);

template <size_t dim>
void augment(const Shape<2>& shape, Tensor<2> &image, Tensor<dim> &res) {
  assert(image.size == res.size);

  zero(res);
  int arg0 = shape[0];
  int arg1 = shape[1];
  size_t yBeg = std::max<int>(0, arg0);
  size_t xBeg = std::max<int>(0, arg1);
  size_t yEnd = std::min<int>(res.ht, res.ht + arg0);
  size_t xEnd = std::min<int>(res.ht, res.ht + arg1);

  for (size_t y = yBeg; y < yEnd; y++) {
    for (size_t x = xBeg; x < xEnd; x++) {
      res.v[(y - arg0) * res.wt + x - arg1] = image.v[y * res.wt + x];
    }
  }
}

class MnistReader {
private:
  Tensor<3> images;
  TensorT<size_t, 1> labels;
  std::string images_path;
  std::string labels_path;

public:
  size_t number_of_entries;
  TensorT<size_t, 1> permutation;
  const Shape<2> augmentations[9] = {{0, 0},  {0, 0}, {1, 0}, {-1, 0}, {0, 1}, {0, -1}};

  int index = -1;
  Matrix last_read;
  int last_label;

  MnistReader() {}
  MnistReader(std::string images_path, std::string labels_path, Shape<2> images_shape, size_t number_of_entries);
  MnistReader(const MnistReader &reader, size_t from, size_t to);
  
  const Tensor<3> &getAllImages() { return images; };
  void loop_to_beg();
  void shuffle();

  template <size_t dim>
  bool read_next(bool do_augmentation, Tensor<dim> &read_to, size_t &label) {
    if (index + 1 >= (int)number_of_entries) return false;

    index++;
    last_label = labels.v[permutation[index]];
    label = last_label;

    last_read.v = images.v + permutation[index]*images.ht*images.wt;
    if (do_augmentation) augment(augmentations[rand() % NAUGMENTATIONS], last_read, read_to);
    else copyToTensorOfSameSize(last_read, read_to);
    return true;
  }
};

#endif
