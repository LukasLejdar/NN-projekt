#ifndef MNIST_READER 
#define MNIST_READER

#include <string>
#include <fstream>
#include "network/math.hpp"

class MnistReader {
  public:
    int magic_number;
    int number_of_entries;
    int height;
    int width;
    int index = -1;

    MnistReader(std::string images_path, std::string labels_path);

    int last_lable;
    Matrix last_read;
    Matrix& read_next();

  private:
    std::ifstream image_file;
    std::ifstream labels_file;
};

#endif
