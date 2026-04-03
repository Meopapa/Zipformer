#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

// --- 1. Hàm Softmax (Giữ nguyên logic của bạn) ---
void softmax(float* input, float* output, int n) {
    float max_val = input[0];
    for (int i = 1; i < n; i++) {
        if (input[i] > max_val) max_val = input[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        output[i] = expf(input[i] - max_val);
        sum += output[i];
    }
    for (int i = 0; i < n; i++) {
        output[i] /= sum;
    }
}

// --- 2. Hàm Downsample (Logic xử lý khối Zipformer) ---
void downsample(float *src, int T, int B, int C, int ds, float *bias, float *output) {
    int dT = (T + ds - 1) / ds;
    int padded_T = dT * ds;
    
    // Cấp phát pad để xử lý trường hợp T không chia hết cho ds
    float *src_pad = (float *)malloc(padded_T * B * C * sizeof(float));
    for (int t = 0; t < padded_T; t++) {
        int src_t = (t < T) ? t : (T - 1);
        for (int b = 0; b < B; b++) {
            memcpy(&src_pad[(t * B + b) * C], &src[(src_t * B + b) * C], C * sizeof(float));
        }
    }

    float *weights = (float *)malloc(ds * sizeof(float));
    softmax(bias, weights, ds);

    for (int dt = 0; dt < dT; dt++) {
        for (int b = 0; b < B; b++) {
            for (int c = 0; c < C; c++) {
                float val = 0.0f;
                for (int i = 0; i < ds; i++) {
                    int t = dt * ds + i;
                    val += src_pad[(t * B + b) * C + c] * weights[i];
                }
                output[(dt * B + b) * C + c] = val;
            }
        }
    }
    free(weights);
    free(src_pad);
}