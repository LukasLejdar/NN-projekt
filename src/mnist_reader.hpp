#ifndef MNIST_READER 
#define MNIST_READER

#include <ios>
#include <iostream>
#include <string>
#include <fstream>
#include "network/math.hpp"

class MnistReader {
  public:
    int number_of_entries;
    int height;
    int width;
    int index = -1;

    MnistReader(std::string images_path, std::string labels_path);

    int last_lable;
    Matrix last_read;
    Matrix& read_next();
    void loop_to_beg();

  private:
    std::ifstream image_file;
    std::ifstream labels_file;
    std::streampos images_begin;
    std::streampos labels_beg;

};

#endif
