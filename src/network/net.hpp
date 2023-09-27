#ifndef NET_H
#define NET_H

struct Dense {
  int in_shape;
  int out_shape;
  float* w;
  float* b;
};

class Net {
  public:
    Dense* layers;
    Net(int in_shapes[], int length);

    void forward_prop(float v[]);
};

void initialize_layer(Dense dense);

#endif
