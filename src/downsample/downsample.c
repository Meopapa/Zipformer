#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// --- 1. Softmax ---
void softmax(double* input, double* output, int n) {
    double max_val = input[0];
    for (int i = 1; i < n; i++)
        if (input[i] > max_val) max_val = input[i];

    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        output[i] = exp(input[i] - max_val);
        sum += output[i];
    }

    for (int i = 0; i < n; i++)
        output[i] /= sum;
}

// --- 2. Downsample ---
void downsample(double *src, int T, int B, int C, int ds, double *bias, double *output) {
    double *weights = (double *)malloc(ds * sizeof(double));
    softmax(bias, weights, ds);

    for (int t = 0; t < T; t++) {
        for (int b = 0; b < B; b++) {
            for (int c = 0; c < C; c++) {
                double sum = 0.0;
                for (int i = 0; i < ds; i++) {
                    int src_t = t + i;
                    if (src_t >= T) src_t = T - 1;

                    // src có hình dạng (T, B, C)
                    sum += src[(src_t * B + b) * C + c] * weights[i];
                }
                // output có hình dạng (T, B, C)
                output[(t * B + b) * C + c] = sum;
            }
        }
    }
    free(weights);
}

// --- 3. Đọc dữ liệu ---
void read_data(const char* filename, double* array, int size) {
    FILE* f = fopen(filename, "r");
    if (!f) {
        printf("Cannot open file %s\n", filename);
        exit(1);
    }

    for (int i = 0; i < size; i++) {
        // Đọc số thực và bỏ qua dấu phẩy nếu có
        if (fscanf(f, " %lf,", &array[i]) != 1) break;
    }

    fclose(f);
}

int main() {
    int T = 202, B = 1, C = 256, ds = 2;
    int total_elements = T * B * C;

    double *src = malloc(total_elements * sizeof(double));
    double *bias = malloc(ds * sizeof(double));
    double *output = malloc(total_elements * sizeof(double));

    read_data("input_0.txt", src, total_elements);
    read_data("encoder_encoders_1_downsample_bias.txt", bias, ds);

    downsample(src, T, B, C, ds, bias, output);

    // --- 4. Ghi file khớp định dạng output.txt ---
    FILE *f_out = fopen("c_result.txt", "w");
    if (!f_out) {
        printf("Cannot create output file\n");
        return 1;
    }

    for (int t = 0; t < T; t++) {
        for (int c = 0; c < C; c++) {
            // Với B=1, chỉ số là t * C + c
            // Đảm bảo truy cập đúng mảng output đã tính ở trên
            int idx = t * C + c; 

            fprintf(f_out, "%.18e", output[idx]);

            // Ghi dấu phẩy giữa các phần tử trong cùng một dòng (kênh C)
            if (c < C - 1) {
                fprintf(f_out, ",");
            }
        }
        // Kết thúc mỗi dòng T bằng một ký tự xuống dòng
        // Không thêm dòng trống ở cuối cùng để khớp chuẩn NumPy/Text
        if (t < T - 1) {
            fprintf(f_out, "\n");
        }
    }

    fclose(f_out);

    printf("DONE! c_result.txt has been created with the correct format.\n");

    free(src);
    free(bias);
    free(output);

    return 0;
}