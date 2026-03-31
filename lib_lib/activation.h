#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "D:\AI_C\lib\tensor.h"

int TENSOR_Sigmoid(Tensor *t);
int TENSOR_Softmax(Tensor *t);
int TENSOR_ReLU(Tensor *t);
int TENSOR_Swoosh(Tensor *t, char type);

#endif 