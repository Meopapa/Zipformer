#ifndef TENSOR_H
#define TENSOR_H

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

typedef struct Tensor
{
    double *data; // Batch, channel, height, width - Batch, seqlen, feature
    int dim1, dim2, dim3, dim4; // Dùng từ dim4 đến dim1
}Tensor;

#define T_SIZE sizeof(Tensor)

int TENSOR_TensorSize(Tensor *tensor);
int TENSOR_Index(int r, int c, int d, int dim4, Tensor *tensor);
int TENSOR_Create(Tensor **tensor, int dim1, int dim2, int dim3, int dim4);
int TENSOR_Init(Tensor **tensor);
int TENSOR_Free(Tensor **tensor);
double TENSOR_ScalarMul(Tensor *tensor_1, Tensor *tensor_2);
Tensor *TENSOR_Matmul(Tensor *tensor_1, Tensor *tensor_2);
int TENSOR_Add(Tensor *tensor_1, Tensor *tensor_2);
Tensor *TENSOR_Sub(Tensor *tensor_1, Tensor *tensor_2);
int TENSOR_Unsqueeze(Tensor *tensor, int dim);
int TENSOR_Transpose(Tensor *tensor, int *dim, int n);
int TENSOR_Reshape(Tensor *tensor, int new_row, int new_collumn, int new_depth, int new_dim);
int TENSOR_Mul(double num, Tensor *tensor);
int TENSOR_Padding(Tensor *tensor, int pad_row, int pad_collumn, int pad_depth);
int TENSOR_conv2d(Tensor *imf, Tensor *omf, Tensor *kernel, Tensor *bias, int stride_h, int stride_w, int pad_h, int pad_w, int groups);
int TENSOR_Linear(Tensor *imf, Tensor *omf, Tensor *weight, Tensor *bias);

#endif