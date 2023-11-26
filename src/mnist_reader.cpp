#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <assert.h>
#include "network/math.hpp"
#include "mnist_reader.hpp"

int reverseInt(int i) {
  unsigned char c1, c2, c3, c4;

  c1 = i & 255;
  c2 = (i >> 8) & 255;
  c3 = (i >> 16) & 255;
  c4 = (i >> 24) & 255;

  return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

MnistReader::MnistReader(std::string images_path, std::string labels_path) : 
  images_path(images_path),
  labels_path(labels_path),
  images_file(images_path), 
  labels_file(labels_path) 
{

  if (!images_file.is_open()) {
    std::cout << "Error opening image_file: " << images_path << "\n";
    return;
  }

  if (!images_file.is_open()) {
    std::cout << "Error opening labels_file" << labels_path << "\n";
    return;
  }

  // read magic numbers ---------------------------

  int magic_number;
  images_file.read((char *)&magic_number, sizeof(magic_number));
  assert(2051 == reverseInt(magic_number));

  labels_file.read((char *)&magic_number, sizeof(magic_number));
  assert(2049 == reverseInt(magic_number));

  // get dimentions -------------------------------

  images_file.read((char *)&number_of_entries, sizeof(number_of_entries));
  number_of_entries = reverseInt(number_of_entries);

  images_file.read((char *)&height, sizeof(height));
  height = reverseInt(height);

  images_file.read((char *)&width, sizeof(width));
  width = reverseInt(width);
  
  std::cout << number_of_entries << " " << height << " " << width << "\n";

  int number_of_labels;
  labels_file.read((char *)&number_of_labels, sizeof(number_of_labels));
  assert(number_of_entries == reverseInt(number_of_labels));

  // set begining positions -----------------------

  images_begin = images_file.tellg();
  labels_begin = labels_file.tellg();

  // intit Matrix ---------------------------------

  last_read = {static_cast<size_t>(height), static_cast<size_t>(width)};
}

MnistReader::MnistReader(MnistReader& other, int from, int to): 
    number_of_entries(to - from),
    height(other.height),
    width(other.width),
    images_path(other.images_path),
    labels_path(other.labels_path),
    images_file(other.images_path), 
    labels_file(other.labels_path),
    images_begin(other.images_begin),
    labels_begin(other.labels_begin)
  {
    assert(0 <= from && from < to);
    assert(to <= other.number_of_entries);

    if (!images_file.is_open()) {
      std::cout << "Error opening image_file: " << images_path << "\n";
      return;
    }

    if (!images_file.is_open()) {
      std::cout << "Error opening labels_file" << labels_path << "\n";
      return;
    }

    last_read = {static_cast<size_t>(height), static_cast<size_t>(width)};
    images_begin += height*width*sizeof(unsigned char)*from;
    labels_begin += sizeof(unsigned char)*from;
    loop_to_beg();

  }

bool MnistReader::read_next(Matrix* saveto, int* lable) {
  if (index >= number_of_entries) return false;

  unsigned char temp = 0;
  labels_file.read((char *)&temp, sizeof(temp));
  last_lable = temp;

  for (int i = 0; i < height*width; ++i) {
      images_file.read((char *)&temp, sizeof(temp));
      last_read.v[i] = temp/255.0;
  }

  if(saveto != nullptr) { copyToTensorOfSameSize(last_read, *saveto); }
  if(lable != nullptr) { *lable = last_lable; };

  index++;
  return true;
}

void MnistReader::loop_to_beg() {
  index = 0;
  images_file.seekg(images_begin);
  labels_file.seekg(labels_begin);
}

