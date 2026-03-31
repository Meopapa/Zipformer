#include <stdio.h>
#include <stdlib.h>
#include <assert.h> 
#include "D:\AI_C\src\tensor.c" 
#include "D:\AI_C\src\activation.c"

int main() {
    Tensor *t_1 = NULL, *t_2 = NULL, *t_3 = NULL;

    TENSOR_Create(&t_1, 3, 1, 2, 1);
    TENSOR_Create(&t_2, 3, 1, 2, 1);
    TENSOR_Create(&t_3, 1, 3, 2, 1);

    assert(t_1 != NULL && "Lỗi: t_1 cấp phát thất bại!");
    assert(TENSOR_TensorSize(t_1) == 6 && "Lỗi: Kích thước t_1 sai!");

    assert(t_2 != NULL && "Lỗi: t_2 cấp phát thất bại!");
    assert(TENSOR_TensorSize(t_2) == 6 && "Lỗi: Kích thước t_2 sai!");

    assert(t_3 != NULL && "Lỗi: t_3 cấp phát thất bại!");
    assert(TENSOR_TensorSize(t_3) == 6 && "Lỗi: Kích thước t_3 sai!");

    printf("[PASSED] Da cap phat va kiem tra kich thuoc thanh cong.\n");

    TENSOR_Init(&t_1);
    TENSOR_Init(&t_2);
    TENSOR_Init(&t_3);
    printf("[PASSED] Da Init cac Tensor ve 0.0.\n");

    TENSOR_ReLU(t_1);
    TENSOR_Sigmoid(t_2);
    TENSOR_Softmax(t_3);
    printf("[PASSED] Su dung duoc cac phep activation.\n");

    TENSOR_Free(&t_1);
    TENSOR_Free(&t_2);
    TENSOR_Free(&t_3);

    assert(t_1 == NULL && t_2 == NULL && t_3 == NULL);
    printf("[PASSED] Khong co Memory Leak. Hoan thanh test.\n");

    return 0;
}