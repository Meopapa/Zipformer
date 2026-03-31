#include "D:\AI_C\lib\tensor.h"

// Create tensor
int TENSOR_Create(Tensor **tensor, int dim1, int dim2, int dim3, int dim4)
{
    if(tensor == NULL || !(dim1 * dim1 + dim2 * dim2 + dim3 * dim3)) return 0;

    // Cấp phát struct Tensor
    *tensor = (Tensor *)malloc(T_SIZE);
    if (*tensor == NULL) return 0;
    
    (*tensor)->dim1 = dim1;
    (*tensor)->dim2 = dim2;
    (*tensor)->dim3 = dim3;
    (*tensor)->dim4 = dim4;
    
    // Cấp phát mảng data 1D
    (*tensor)->data = (double *)malloc(TENSOR_TensorSize(*tensor) * sizeof(double));
    if ((*tensor)->data == NULL) {
        free(*tensor); // Chống rò rỉ nếu malloc thứ 2 thất bại
        *tensor = NULL;
        return 0;
    }
    return 1;
}

int TENSOR_Init(Tensor **tensor)
{
    if(tensor == NULL || *tensor == NULL) return 0;

    for(int i = 0; i < TENSOR_TensorSize(*tensor); i++)
        (*tensor)->data[i] = 0.0;
    
    return 1;
}

int TENSOR_Free(Tensor **tensor)
{
    if (tensor == NULL || *tensor == NULL) return 0;
    
    free((*tensor)->data); // Giải phóng mảng trước
    free(*tensor);         // Giải phóng struct sau
    *tensor = NULL;        // Tránh lỗi Use-After-Free
    return 1;
}

// Location
int TENSOR_Index(int n, int c, int h, int w, Tensor *t) 
{
    return ((n * t->dim2 + c) * t->dim3 + h) * t->dim4 + w;
}

int TENSOR_TensorSize(Tensor *tensor)
{
    return tensor->dim1 * tensor->dim2 * tensor->dim3 * tensor->dim4;
}

// Operation
double TENSOR_ScalarMul(Tensor *tensor_1, Tensor *tensor_2)
{
    if(tensor_1 == NULL || tensor_2 == NULL) return 0.0;
    if(TENSOR_TensorSize(tensor_1) != TENSOR_TensorSize(tensor_2)) return 0.0;
    if(tensor_1->dim1 != tensor_2->dim1 || tensor_1->dim2 != tensor_2->dim2 || tensor_1->dim3 != tensor_2->dim3 || tensor_1->dim4 != tensor_2->dim4) return 0.0;

    double sum;
    
    sum = 0.0;
    for(int i = 0; i < TENSOR_TensorSize(tensor_1); i++)
        sum += tensor_1->data[i] * tensor_2->data[i];

    return sum;
}

Tensor *TENSOR_Matmul(Tensor *tensor_1, Tensor *tensor_2)
{
    if(tensor_1 == NULL || tensor_2 == NULL) return NULL;
    if(tensor_1->dim2 != tensor_2->dim1 || tensor_1->dim3 != tensor_2->dim3) return NULL;

    Tensor *t = NULL;
    double num = 0.0;

    if (!TENSOR_Create(&t, tensor_1->dim1, tensor_2->dim2, tensor_1->dim3, 1)) return NULL;
    TENSOR_Init(&t);

    for(int m = 0; m < tensor_1->dim3; m++)
    {
        for(int i = 0; i < tensor_1->dim1; i++)
        {
            for(int k = 0; k < tensor_2->dim1; k++)
            {
                num = tensor_1->data[TENSOR_Index(i,k,m,1,tensor_1)];
                for(int j = 0; j < tensor_2->dim2; j++)
                {
                    t->data[TENSOR_Index(i,j,m,1,t)] += num * tensor_2->data[TENSOR_Index(k,j,m,1,tensor_2)];
                }
            }
        }
    }

    return t;
}

int TENSOR_Add(Tensor *dest, Tensor *src)
{
    if(dest == NULL || src == NULL) return 0;
    if(dest->dim1 != src->dim1 || dest->dim2 != src->dim2 || 
       dest->dim3 != src->dim3 || dest->dim4 != src->dim4) return 0;

    int size; 
    size = TENSOR_TensorSize(dest);
    for(int i = 0; i < size; i++) 
    {
        dest->data[i] += src->data[i];
    }
    
    return 1;
}

Tensor *TENSOR_Sub(Tensor *tensor_1, Tensor *tensor_2)
{
    if(tensor_1 == NULL || tensor_2 == NULL) return NULL;
    if(tensor_1->dim1 != tensor_2->dim1 || tensor_1->dim2 != tensor_2->dim2 || tensor_1->dim3 != tensor_2->dim3 || tensor_1->dim4 != tensor_2->dim4) return NULL;

    Tensor* t = NULL;

    if (!TENSOR_Create(&t, tensor_1->dim1, tensor_1->dim2, tensor_1->dim3, tensor_1->dim4)) return NULL;
    TENSOR_Init(&t);

    for(int i = 0; i < TENSOR_TensorSize(tensor_1); i++) t->data[i] = tensor_1->data[i] - tensor_2->data[i];
    
    return t;
}

int TENSOR_Transpose(Tensor *tensor, int *dimension, int n) {
    assert(tensor && "Pointer error");

    Tensor *old = NULL;
    TENSOR_Create(&old, tensor->dim1, tensor->dim2, tensor->dim3, tensor->dim4);
    for(int i = 0; i < TENSOR_TensorSize(tensor); i++) old->data[i] = tensor->data[i];

    int dims[4] = {old->dim1, old->dim2, old->dim3, old->dim4};
    int new_dims[4];
    for(int i = 0; i < 4; i++) new_dims[i] = dims[dimension[i]];

    tensor->dim1 = new_dims[0];
    tensor->dim2 = new_dims[1];
    tensor->dim3 = new_dims[2];
    tensor->dim4 = new_dims[3];

    for (int i1 = 0; i1 < tensor->dim1; i1++) {
        for (int i2 = 0; i2 < tensor->dim2; i2++) {
            for (int i3 = 0; i3 < tensor->dim3; i3++) {
                for (int i4 = 0; i4 < tensor->dim4; i4++) {
                    
                    int old_idx[4];
                    int new_idx[4] = {i1, i2, i3, i4};
                    for(int d = 0; d < 4; d++) old_idx[dimension[d]] = new_idx[d];

                    double val = old->data[TENSOR_Index(old_idx[0], old_idx[1], old_idx[2], old_idx[3], old)];
                    tensor->data[TENSOR_Index(i1, i2, i3, i4, tensor)] = val;
                }
            }
        }
    }

    TENSOR_Free(&old);
    return 1;
}

int TENSOR_Reshape(Tensor *tensor, int new_row, int new_collumn, int new_depth, int new_dim)
{
    if(tensor == NULL) return 0;
    if(!(tensor->dim1 && tensor->dim2 && tensor->dim3 && tensor->dim4)) return 0;

    tensor->dim1 = new_row;
    tensor->dim2 = new_collumn;
    tensor->dim3 = new_depth;
    tensor->dim4 = new_dim;

    return 1;
}

int TENSOR_Mul(double num, Tensor *tensor)
{
    if(tensor == NULL) return 0;
    if(!(tensor->dim1 && tensor->dim2 && tensor->dim3 && tensor->dim4)) return 0;

    if(num == 1) return 1;

    for(int i = 0; i < TENSOR_TensorSize(tensor); i++) tensor->data[i] *= num;

    return 0;
}

int TENSOR_Unsqueeze(Tensor *tensor, int dim)
{
    assert(tensor && "Pointer error");
    assert(dim >= 0 && dim <= 3 && "Dim out of range for 4D struct");
    
    switch(dim)
    {
        case 0: 
            tensor->dim1 = 1; 
            break;
        case 1: 
            tensor->dim1 = tensor->dim2;
            tensor->dim2 = 1;
            break;
        case 2:
            tensor->dim1 = tensor->dim2;
            tensor->dim2 = tensor->dim3;
            tensor->dim3 = 1;
            break;
        case 3:
            tensor->dim1 = tensor->dim2;
            tensor->dim2 = tensor->dim3;
            tensor->dim3 = tensor->dim4;
            tensor->dim4 = 1;
            break;
    }
    return 1;
}

/**
 * @brief Hàm Conv2d tổng quát hỗ trợ Groups (Depthwise, Pointwise, Normal)
 * * @param input     Dữ liệu vào [N, C_in, H_in, W_in]
 * @param weight    Trọng số [C_out, C_in/groups, K_h, K_w]
 * @param bias      Độ lệch [C_out] (có thể NULL)
 * @param output    Kết quả [N, C_out, H_out, W_out]
 * @param groups    Số nhóm (1 cho Normal, C_in cho Depthwise)
 */
int TENSOR_conv2d(Tensor *imf, Tensor *omf, Tensor *kernel, Tensor *bias, int stride_h, int stride_w, int pad_h, int pad_w, int groups) 
{
    // Kiểm tra con trỏ
    assert(imf && omf && kernel && bias && "Pointer error");
    assert(stride_h > 0 && stride_w > 0 && "Stride error");
    assert((imf->dim2 % groups == 0) && (omf->dim2 % groups == 0) && "Invalid groups");
    assert(!(kernel->dim2 - imf->dim2/groups || omf->dim2 - bias->dim1 || omf->dim2 - kernel->dim1 || omf->dim1 - imf->dim1) && "Invalid dimension");
    assert(omf->dim3 == (imf->dim3 + 2 * pad_h - kernel->dim3) / stride_h + 1 && "Output height error");
    assert(omf->dim4 == (imf->dim4 + 2 * pad_w - kernel->dim4) / stride_w + 1 && "Output width error");

    int in_channels_per_group = imf->dim2 / groups;
    int out_channels_per_group = omf->dim2 / groups;

    for(int n = 0; n < omf->dim1; n++) {
        for(int g = 0; g < groups; g++) {
            for(int c_o = 0; c_o < out_channels_per_group; c_o++) {
                int c_out = g * out_channels_per_group + c_o; 
                double b_val = bias->data[c_out]; // Lấy bias một lần cho toàn bộ output map

                for(int h_out = 0; h_out < omf->dim3; h_out++) {
                    for(int w_out = 0; w_out < omf->dim4; w_out++) {
                        double sum = b_val; 

                        for(int c_i = 0; c_i < in_channels_per_group; c_i++) {
                            int c_in = g * in_channels_per_group + c_i; 

                            for(int h_k = 0; h_k < kernel->dim3; h_k++) {
                                int h_in = h_out * stride_h - pad_h + h_k;
                                if (h_in < 0 || h_in >= imf->dim3) continue; // Skip padding nhanh

                                for(int w_k = 0; w_k < kernel->dim4; w_k++) {
                                    int w_in = w_out * stride_w - pad_w + w_k;
                                    
                                    if (w_in >= 0 && w_in < imf->dim4) {
                                        // Sử dụng c_i thay vì c_in cho trọng số của group
                                        double input_val = imf->data[TENSOR_Index(n, c_in, h_in, w_in, imf)];
                                        double weight_val = kernel->data[TENSOR_Index(c_out, c_i, h_k, w_k, kernel)];
                                        sum += input_val * weight_val;
                                    }
                                }
                            }
                        }
                        omf->data[TENSOR_Index(n, c_out, h_out, w_out, omf)] = sum;
                    }
                }
            }
        }
    }
    return 1;
}

/*
** @brief Code nn.Linear
*/
int TENSOR_Linear(Tensor *imf, Tensor *omf, Tensor *weight, Tensor *bias)
{
    assert(imf && omf && weight && "Pointer error");
    assert(imf->dim4 == weight->dim4 && omf->dim4 == weight->dim3 && omf->dim1 == imf->dim1 && omf->dim2 == imf->dim2 && omf->dim3 == imf->dim3 && weight->dim1 == 1 && weight->dim2 == 1 && "Invalid dimension");

    if(bias) assert(omf->dim4 == bias->dim4 && bias->dim2 == 1 && bias->dim3 == 1 && bias->dim1 == 1 && "Bias dimension error");
    
/*
    // for(int o_1 = 0; o_1 < omf->dim1; o_1++) 
    // {
    //     for(int o_2 = 0; o_2 < omf->dim2; o_2++)
    //     {
    //         for(int o_3 = 0; o_3 < omf->dim3; o_3++)
    //         {
    //             for(int o_4 = 0; o_4 < omf->dim4; o_4++)
    //             {
    //                 double sum = bias ? bias->data[o_4] : 0.0;
    //                 for(int i_4 = 0; i_4 < imf->dim4; i_4++)
    //                 {
    //                     double input_val = imf->data[TENSOR_Index(o_1, o_2, o_3, i_4, imf)];
    //                     double weight_val = weight->data[TENSOR_Index(0, 0, o_4, i_4, weight)];
    //                     sum += input_val * weight_val;
    //                 }
    //                 omf->data[TENSOR_Index(o_1, o_2, o_3, o_4, omf)] = sum;
    //             }
    //         }
    //     }
    // }
*/

    int size = imf->dim1 * imf->dim2 * imf->dim3; 
    int in_features = imf->dim4;
    int out_features = omf->dim4;

    for (int s = 0; s < size; s++) 
    {
        for (int o_f = 0; o_f < out_features; o_f++) 
        {
            double sum = bias ? bias->data[o_f] : 0.0;
            
            for (int i_f = 0; i_f < in_features; i_f++) 
            {
                
                double input_val = imf->data[s * in_features + i_f];
                
                double weight_val = weight->data[TENSOR_Index(0, 0, o_f, i_f, weight)];
                
                sum += input_val * weight_val;
            }
            
            omf->data[s * out_features + o_f] = sum;
        }
    }
}