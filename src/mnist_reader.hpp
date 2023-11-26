#ifndef MNIST_READER 
#define MNIST_READER

#include <cstddef>
#include <ios>
#include <iostream>
#include <mutex>
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
    MnistReader(MnistReader& reader, int from, int to);

    int last_lable;
    Matrix last_read;
    bool read_next(Matrix* saveto = nullptr, int* lable = nullptr);
    void loop_to_beg();
    void swap(const MnistReader& other);

  private:
    std::string images_path;
    std::string labels_path;
    std::ifstream images_file;
    std::ifstream labels_file;
    std::streampos images_begin;
    std::streampos labels_begin;

};

#endif
