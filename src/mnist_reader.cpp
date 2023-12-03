#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <assert.h>
#include "network/math.hpp"
#include "mnist_reader.hpp"
#include <numeric>

MnistReader::MnistReader(std::string images_path, std::string labels_path, Shape<2> shape, size_t number_of_entries) : 
  images(readMnistImagesCsv(images_path, shape, number_of_entries)),
  labels(readMnistLablesCsv(labels_path, number_of_entries)),
  images_path(images_path),
  labels_path(labels_path),
  number_of_entries(number_of_entries),
  permutation(number_of_entries),
  last_read(images[0]),
  last_label(labels[0])
{
  srand(time(NULL));
  std::iota(permutation.v, permutation.v + permutation.size, 0);
  loop_to_beg();
}

MnistReader::MnistReader(const MnistReader& other, size_t from, size_t to): 
  images(other.images.reference()),
  labels(other.labels.reference()),
  images_path(other.images_path),
  labels_path(other.labels_path),
  number_of_entries(to - from),
  permutation(other.permutation.reference(from, to)),
  last_read(images[permutation[0]]),
  last_label(labels[permutation[0]])
{
  loop_to_beg();
}

void MnistReader::loop_to_beg() {
  index = -1;
}

void MnistReader::shuffle() {
  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(permutation.beg(), permutation.end(), g);
}

Tensor<3>& readMnistImagesCsv(std::string file_path, Shape<2> shape, size_t number_of_entries) {
  std::ifstream images_file(file_path);

  if (!images_file.is_open()) {
    throw "Error opening Training csv";
  }

  Tensor<3>* images = new Tensor<3>(number_of_entries, shape[0], shape[1]);
  size_t digit;
  char temp;

  for(size_t i = 0; i < number_of_entries; i++) {
    for(size_t j = 0; j < shape.size; j++) {
      digit = 0;
      char delimiter = (j == shape.size-1) ? 0x0a : ',';

      for(size_t k = 0; k < 4; k++) {
        images_file.get(temp);
        if(temp == delimiter) {
          images->v[i*shape.size + j] = digit / 255.0f;
          break;
        }
        assert(k < 3 && "More than 3 chars between commas in an images file");
        digit = digit*10 + temp - '0';
      }
    }
  }

  return *images;
};

TensorT<size_t, 1>& readMnistLablesCsv(std::string file_path, size_t number_of_entries) {
  std::ifstream labels_file(file_path);

  if (!labels_file.is_open()) {
    throw "Error opening Lables csv";
  }

  TensorT<size_t, 1>* lables = new TensorT<size_t, 1>(number_of_entries);
  for(size_t i = 0; i < number_of_entries; i++) {
    char temp = 0;
    labels_file.get(temp);
    lables->v[i] = temp - '0';
    labels_file.get(temp);
    assert(temp == 0x0a && "More than a single char on line in a labels file");
  }

  return *lables;
}
