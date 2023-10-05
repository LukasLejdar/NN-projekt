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

MnistReader::MnistReader(std::string image_path, std::string labels_path) : image_file(image_path), labels_file(labels_path) {

  if (!image_file.is_open()) {
    std::cout << "Error opening image_file: " << image_path << "\n";
    return;
  }

  if (!image_file.is_open()) {
    std::cout << "Error opening labels_file" << labels_path << "\n";
    return;
  }

  int magic_number;
  image_file.read((char *)&magic_number, sizeof(magic_number));
  assert(2051 == reverseInt(magic_number));

  labels_file.read((char *)&magic_number, sizeof(magic_number));
  assert(2049 == reverseInt(magic_number));

  image_file.read((char *)&number_of_entries, sizeof(number_of_entries));
  number_of_entries = reverseInt(number_of_entries);

  image_file.read((char *)&height, sizeof(height));
  height = reverseInt(height);

  image_file.read((char *)&width, sizeof(width));
  width = reverseInt(width);

  std::cout << number_of_entries << " " << height << " " << width << "\n";
  
  int number_of_labels;
  labels_file.read((char *)&number_of_labels, sizeof(number_of_labels));
  assert(number_of_entries == reverseInt(number_of_labels));

  float* v = new float[height*width];
  memset(v, 0, height*width*sizeof(float));
  last_read = {height, width, v};
}

Matrix& MnistReader::read_next() {
  unsigned char temp = 0;
  labels_file.read((char *)&temp, sizeof(temp));
  last_lable = temp;

  for (int i = 0; i < height*width; ++i) {
      image_file.read((char *)&temp, sizeof(temp));
      last_read.v[i] = temp/255.0;
  }
  index++;
  return last_read;
}
