#include "D:\AI_C\lib\activation.h"

int TENSOR_Sigmoid(Tensor *t)
{
    assert(t != NULL && "tensor is null");
    assert((t->dim1 && t->dim2 && t->dim3 && t->dim4) && "tensor dimension invalid");

    int n = t->dim1 * t->dim2 * t->dim3 * t->dim4;

    for(int i = 0; i < n; i++) t->data[i] = 1.0/(1.0 + exp(-t->data[i]));

    return 1;
}

/** 
 * @brief version of softmax just for 3 dimension tensor(dim1,dim2,dim3)
*/
int TENSOR_Softmax(Tensor *t) {
    assert(t != NULL && "tensor is null");
    assert((t->dim1 > 0 && t->dim2 > 0 && t->dim3 > 0) && "tensor dimension invalid");

    for(int i = 0; i < t->dim3; i++) {
        for(int j = 0; j < t->dim1; j++) {
            
            int offset = TENSOR_Index(j, 0, i, 1, t);
            
            double max = t->data[offset];
            for(int k = 1; k < t->dim2; k++) {
                if(t->data[offset + k] > max) {
                    max = t->data[offset + k];
                }
            }

            double sum = 0.0;
            for(int k = 0; k < t->dim2; k++) {
                t->data[offset + k] = exp(t->data[offset + k] - max);
                sum += t->data[offset + k];
            }

            for(int k = 0; k < t->dim2; k++) {
                t->data[offset + k] /= sum;
            }
        }
    }
    return 1;
}

int TENSOR_ReLU(Tensor *t)
{
    assert(t != NULL && "tensor is null");
    assert((t->dim1 && t->dim2 && t->dim3 && t->dim4) && "tensor dimension invalid");

    int n = t->dim1 * t->dim2 * t->dim3 * t->dim4;

    for(int i = 0; i < n; i++) if(t->data[i] < 0.0) t->data[i] = 0.0;

    return 1;
}

int TENSOR_Swoosh(Tensor *t, char type)
{
    assert(t && "Pointer error");
    assert(!(type-108 && type-114 ) && "None type 'r' or 'l'");

    double offset, linear, block;

    if(!(type - 108))
    {
        offset = 4.0;
        linear = 0.08;
        block = 0.035;
    }
    if(!(type - 114))
    {
        offset = 1.0;
        linear = 0.08;
        block = 0.313261687;
    }

    for(int i = 0; i < TENSOR_TensorSize(t); i++) t->data[i] = log(1+exp(t->data[i] - offset)) - linear*t->data[i] - block;

    return 1;
}