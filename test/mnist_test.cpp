
#include "../src/mnist_reader.hpp";

int main() {
  //fashion_mnist_test_labels.csv
  //fashion_mnist_test_vectors.csv
  //fashion_mnist_train_labels.csv
  //fashion_mnist_train_vectors.csv
  
  MnistReader training_data("fashion_mnist_train_vectors.csv", "fashion_mnist_train_labels.csv", 60000);
  training_data.read_next();


}
