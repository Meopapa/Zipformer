#include "D:\AI_C\lib\file.h"

int FILE_ReadTensorBinary(Tensor *t, char *path)
{
    assert(t && path && "Pointer error");
    
    FILE *f = fopen(path, "rb");
    int num_elements;
    size_t numofindex;

    assert(f && "Can't open file");

    fseek(f, 0, SEEK_END);
    long long file_size = ftell(f);
    rewind(f);

    num_elements = file_size / sizeof(double);

    assert(num_elements == TENSOR_TensorSize(t) && "Wrong file");

    numofindex = fread(t->data, sizeof(double), num_elements, f);

    assert(numofindex == num_elements && "Read file error");

    fclose(f);

    return 1;
}

int FILE_WriteTensor(){}

int FILE_Read(){}

int FILE_Write(){}