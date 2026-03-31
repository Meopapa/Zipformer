#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include "D:\AI_C\lib\tensor.h"

int read(Tensor *x, Tensor *y, char path[]);
Tensor *inference(Tensor *x, Tensor *w, Tensor *b);
double loss(Tensor *y_pred, Tensor *y_true);
int grad(Tensor *w, Tensor *b, Tensor *x, Tensor *y_true, double learning_rate, int epoch);

int main()
{
    Tensor *x, *y, *w, *b;
}

int read(Tensor *x, Tensor *y, char path[])
{
    assert((x&&y) && "Pointer error");
    assert((len(path)) && "Path error");

    FILE *file = fopen(path, 'r');

    assert((file) && "File error");

    char line[256];
    int row = 0, collumn = 0;

    fgets(line, sizeof(line), file);
    char *token = strtok(line, ",");
    while (token != NULL) {
        collumn++;
        token = strtok(NULL, ",");
    }

    while (fgets(line, sizeof(line), file)) {
        row++;
    }

    TENSOR_Create(&x, row, collumn - 1, 1, 1);
    TENSOR_Create(&y, row, 1, 1, 1);

    while(fgets(line, sizeof(line), file))
    {
        char *token = strtok(line, ",");
        for(int i = 0; i < collumn - 1; i++) {
            x->data[TENSOR_Index(row, i, 0, 1, x)] = atof(token);
            token = strtok(NULL, ",");
        }
        y->data[TENSOR_Index(row, 0, 0, 1, y)] = atof(token);
    }

    fclose(file);
    return 1;
}

Tensor *inference(Tensor *x, Tensor *w, Tensor *b)
{
    assert((x && w && b) && "Pointer error");
    assert((x->dim2 == w->dim1) && "Dimension error");

    Tensor *y_pred = TENSOR_Add(TENSOR_Matmul(x, w), b);
    return y_pred;
}

double loss(Tensor *y_pred, Tensor *y_true)
{
    assert((y_pred && y_true) && "Pointer error");
    assert((y_pred->dim1 == y_true->dim1) && "Dimension error");

    double sum = 0.0;
    for(int i = 0; i < y_pred->dim1; i++) {
        double diff = y_pred->data[TENSOR_Index(i, 0, 0, 1, y_pred)] - y_true->data[TENSOR_Index(i, 0, 0, 1, y_true)];
        sum += diff * diff;
    }
    return sum / (2.0 * y_pred->dim1);
}

int grad(Tensor *w, Tensor *b, Tensor *x, Tensor *y_true, double learning_rate, int epoch)
{
    assert((w && b && x && y_true) && "Pointer error");
    assert((x->dim2 == w->dim1) && "Dimension error");

    for(int e = 0; e < epoch; e++) {
        Tensor *y_pred = inference(x, w, b);
        double current_loss = loss(y_pred, y_true);
        printf("Epoch %d: Loss = %f\n", e + 1, current_loss);

        Tensor *dw = TENSOR_Matmul(TENSOR_Transpose(x, (int[]){x->dim2, x->dim1, x->dim3}, 3), TENSOR_Sub(y_pred, y_true));
        Tensor *db = TENSOR_Sub(y_pred, y_true);

        // for(int i = 0; i < w->row * w->collumn; i++) w->data[i] -= learning_rate * dw->data[i];
        // for(int i = 0; i < b->row * b->collumn; i++) b->data[i] -= learning_rate * db->data[i];

        TENSOR_Sub(w, TENSOR_Mul(learning_rate, dw));
        TENSOR_Sub(b, TENSOR_Mul(learning_rate, db));

        TENSOR_Free(&y_pred);
        TENSOR_Free(&dw);
        TENSOR_Free(&db);
    }
    return 1;
}
