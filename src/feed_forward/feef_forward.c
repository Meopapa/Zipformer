#include<stdio.h>
#include<stdlib.h>
#include<math.h>
// Swoosh_L = log(1 + exp(x-4))-0.08x-0.035
float swoosh_l(float x){
    float softplus= logf(1+expf(x-4.0f));
    return softplus -(0.08f*x) -0.035f;

}
// Hàm Linear: Out = In * W^T + Bias
// in_dim: embed_dim, out_dim: feedforward_dim
void linear(float* input, float* weight, float* bias, float* output
, int in_dim, int out_dim){
    for(int i=0; i<out_dim;i++){
        float sum= bias[i];
        for(int j=0;j<in_dim;j++){
            sum += input[j]*weight[i*in_dim+j];
        }
        output[i] =sum;
    }
}
void feedforward(float* x,
                float* w_in, float* b_in,
                float* w_out, float* b_out,
                float* output,
                int embed_dim, int ff_dim, float initial_scale){
    //cap phat bo nho tam cho lop an(ff_dim)
    float* hidden=(float*)malloc(ff_dim*sizeof(float));
    // 2. Linear In: x -> hidden (in_proj)
    linear(x, w_in, b_in, hidden, embed_dim, ff_dim);
    // 3. Activation: SwooshL (Áp dụng cho từng phần tử trong hidden)
    for (int i = 0; i < ff_dim; i++) {
        hidden[i] = swoosh_l(hidden[i]);
    }
    // 4. Linear Out: hidden -> output (out_proj)
    linear(hidden, w_out, b_out, output, ff_dim, embed_dim);

    // 5. Scaling: Nhân với initial_scale (0.1)
    for (int i = 0; i < embed_dim; i++) {
        output[i] *= initial_scale;
    }

    free(hidden);
    }  