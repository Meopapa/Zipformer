#include <stdio.h>
#include "D:\AI_C\src\tensor.c"

// Hàm bổ trợ in Tensor 4D
void TENSOR_Print4D(const char* name, Tensor *t) {
    printf("Tensor: %s (Shape: %d, %d, %d, %d)\n", name, t->dim1, t->dim2, t->dim3, t->dim4);
    for (int b = 0; b < t->dim1; b++) {
        printf("dim1 %d:\n", b);
        for (int c = 0; c < t->dim2; c++) {
            printf("  dim2 %d:\n", c);
            for (int h = 0; h < t->dim3; h++) {
                printf("    ");
                for (int w = 0; w < t->dim4; w++) {
                    // Dùng TENSOR_Index để đảm bảo ánh xạ đúng vị trí
                    int index = TENSOR_Index(b, c, h, w, t);
                    printf("%8.2f ", t->data[index]);
                }
                printf("\n");
            }
        }
    }
    printf("==========================================\n");
}

int main()
{
    Tensor *imf;
    Tensor *omf;
    Tensor *weight;
    Tensor *bias;
    int p, s, g;

    p = 1;
    s = 1;
    g = 2;

    // Khởi tạo các Tensor
    TENSOR_Create(&imf, 1, 2, 3, 3);
    TENSOR_Create(&omf, 1, 4, 3, 3);
    TENSOR_Create(&weight, 4, 1, 3, 3);
    TENSOR_Create(&bias, 4, 1, 1, 1);

    // Gán giá trị mặc định
    for(int i = 0; i < TENSOR_TensorSize(imf); i++) imf->data[i] = 1.0;
    for(int i = 0; i < TENSOR_TensorSize(weight); i++) weight->data[i] = 1.0;
    for(int i = 0; i < TENSOR_TensorSize(bias); i++) bias->data[i] = 1.0;

    // Thực hiện phép toán Conv2D
    TENSOR_conv2d(imf, omf, weight, bias, s, p, g);

    // In kết quả theo định dạng 4D
    TENSOR_Print4D("Input (imf)", imf);
    TENSOR_Print4D("Weight", weight);
    TENSOR_Print4D("Output (omf)", omf);

    return 0;
}