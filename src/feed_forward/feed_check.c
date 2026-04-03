#include <stdio.h>
#include <stdlib.h>

// Hàm Linear chuẩn của bạn
void linear(float* input, float* weight, float* bias, float* output, int in_dim, int out_dim) {
    for (int i = 0; i < out_dim; i++) {
        float sum = bias[i];
        for (int j = 0; j < in_dim; j++) {
            sum += input[j] * weight[i * in_dim + j];
        }
        output[i] = sum;
    }
}

int main() {
    int in_dim = 640;   
    int out_dim = 192;  
    int T = 404; 

    float* input_frame = (float*)malloc(in_dim * sizeof(float));
    float* weight = (float*)malloc(out_dim * in_dim * sizeof(float));
    float* bias = (float*)malloc(out_dim * sizeof(float));
    float* output = (float*)malloc(out_dim * sizeof(float));

    FILE *fi = fopen("input_feed_out_0.txt", "r");
    FILE *fw = fopen("weight_out.txt", "r");
    FILE *fb = fopen("bias_out.txt", "r");
    FILE *fo = fopen("ket_qua_c.txt", "w"); // Ghi ra file để dễ kiểm tra

    if (!fi || !fw || !fb || !fo) {
        printf("Loi mo file!\n");
        return 1;
    }

    // Nạp Weight và Bias
    for (int i = 0; i < out_dim * in_dim; i++) fscanf(fw, "%f", &weight[i]);
    for (int i = 0; i < out_dim; i++) fscanf(fb, "%f", &bias[i]);
    fclose(fw); fclose(fb);

    // VÒNG LẶP XỬ LÝ 404 ROWS
    for (int t = 0; t < T; t++) {
        
        // 1. Đọc dữ liệu cho 1 frame
        for (int j = 0; j < in_dim; j++) {
            if (fscanf(fi, "%f", &input_frame[j]) != 1) break;
        }

        // 2. Tính toán Linear
        linear(input_frame, weight, bias, output, in_dim, out_dim);

        // 3. IN TOÀN BỘ OUTPUT CỦA 1 FRAME TRÊN 1 HÀNG
        for (int i = 0; i < out_dim; i++) {
            fprintf(fo, "%.6f ", output[i]); // Ghi vào file
            // printf("%.6f ", output[i]);   // Nếu muốn in ra màn hình thì dùng lệnh này
        }
        
        // SAU KHI IN HẾT 192 SỐ THÌ MỚI XUỐNG DÒNG
        fprintf(fo, "\n"); 
        // printf("\n"); 
    }

    fclose(fi);
    fclose(fo);
    free(input_frame); free(weight); free(bias); free(output);
    
    printf("Hoan thanh! Da ghi 404 hang vao file ket_qua_c.txt\n");
    return 0;
}