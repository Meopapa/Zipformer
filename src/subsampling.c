#include "D:\AI_C\src\tensor.c"
#include "D:\AI_C\src\file.c"
#include "D:\AI_C\src\activation.c"

/* Metadata:
Input: 1, 313, 80

Parameters:
encoder_embed_conv_0_weight=8,1,3,3
encoder_embed_conv_0_bias=8
encoder_embed_conv_4_weight=32,8,3,3
encoder_embed_conv_4_bias=32
encoder_embed_conv_7_weight=128,32,3,3
encoder_embed_conv_7_bias=128
encoder_embed_convnext_depthwise_conv_weight=128,1,7,7
encoder_embed_convnext_depthwise_conv_bias=128
encoder_embed_convnext_pointwise_conv1_weight=384,128,1,1
encoder_embed_convnext_pointwise_conv1_bias=384
encoder_embed_convnext_pointwise_conv2_weight=128,384,1,1
encoder_embed_convnext_pointwise_conv2_bias=128
encoder_embed_out_weight=192,2432
encoder_embed_out_bias=192
encoder_embed_out_norm_log_scale=1
encoder_embed_out_norm_bias=192

Output: 
encoder_embed_conv_0|double|1,8,311,80
encoder_embed_conv_1|double|1,8,311,80
encoder_embed_conv_3|double|1,8,311,80
encoder_embed_conv_4|double|1,32,155,39
encoder_embed_conv_6|double|1,32,155,39
encoder_embed_conv_7|double|1,128,153,19
encoder_embed_conv_9|double|1,128,153,19
encoder_embed_convnext_depthwise_conv|double|1,128,153,19
encoder_embed_convnext_pointwise_conv1|double|1,384,153,19
encoder_embed_convnext_swooshL|double|1,384,153,19
encoder_embed_convnext_pointwise_conv2|double|1,128,153,19
encoder_embed_out|double|1,153,192
encoder_embed_out_norm|double|1,153,192
*/

double read_parameters(const char *path)
{
    assert(path && "Path error");

    double data;

    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Error: Failed to open parameter file %s\n", path);
        return 0;
    }

    fread(&data, sizeof(double), 1, f);
    fclose(f);
    return data;
}

int biasnorm(Tensor *x, Tensor *bias, double log_scale)
{
    assert(x && bias && "Pointer error");
    assert(x->dim4 == bias->dim1 || x->dim4 == bias->dim4); 

    int outer_size = x->dim1 * x->dim2 * x->dim3; 
    int num_channels = x->dim4; 
    double eps = 1e-10;
    double exp_log_scale = exp(log_scale);

    for (int i = 0; i < outer_size; i++) {
        int offset = i * num_channels;
        double sum_sq_diff = 0.0;

        for (int c = 0; c < num_channels; c++) {
            double diff = x->data[offset + c] - bias->data[c];
            sum_sq_diff += diff * diff;
        }
        
        double mean_sq_diff = sum_sq_diff / num_channels;
        
        double inv_std = 1.0 / sqrt(mean_sq_diff + eps);
        double final_scale = inv_std * exp_log_scale;

        for (int c = 0; c < num_channels; c++) {
            x->data[offset + c] *= final_scale;
        }
    }

    return 1;
}

void Check_Intermediate(Tensor *tensor_to_check, char *python_bin_path) {
    Tensor *py_tensor;
    TENSOR_Create(&py_tensor, tensor_to_check->dim1, tensor_to_check->dim2, tensor_to_check->dim3, tensor_to_check->dim4);
    FILE_ReadTensorBinary(py_tensor, python_bin_path);
    
    printf("Checking %s...\n", python_bin_path);

    int n = 0;
    for(int i = 0; i < TENSOR_TensorSize(tensor_to_check); i++) {
        if (fabs(tensor_to_check->data[i] - py_tensor->data[i]) > 1e-4 && n < 5) {
            printf("                    [LỖI] Mismatch at index %d: Python %f, C %f\n", i, py_tensor->data[i], tensor_to_check->data[i]);
            n++;
        }
    }
    if (n == 0) printf("                    [PASS] %s khop hoan toan!\n", python_bin_path);
    TENSOR_Free(&py_tensor);
}

int main()
{
    int channel, feature_dim, in_channels;
    double log_scale;

    char *input_file = "..\\docs\\input_features_bin\\input_features.bin"; 

    char *param_files[] = {
        "..\\docs\\model_weight_bin\\encoder_embed_conv_0_weight.bin",
        "..\\docs\\model_weight_bin\\encoder_embed_conv_0_bias.bin",
        "..\\docs\\model_weight_bin\\encoder_embed_conv_4_weight.bin",
        "..\\docs\\model_weight_bin\\encoder_embed_conv_4_bias.bin",
        "..\\docs\\model_weight_bin\\encoder_embed_conv_7_weight.bin",
        "..\\docs\\model_weight_bin\\encoder_embed_conv_7_bias.bin",
        "..\\docs\\model_weight_bin\\encoder_embed_convnext_depthwise_conv_weight.bin",
        "..\\docs\\model_weight_bin\\encoder_embed_convnext_depthwise_conv_bias.bin",
        "..\\docs\\model_weight_bin\\encoder_embed_convnext_pointwise_conv1_weight.bin",
        "..\\docs\\model_weight_bin\\encoder_embed_convnext_pointwise_conv1_bias.bin",
        "..\\docs\\model_weight_bin\\encoder_embed_convnext_pointwise_conv2_weight.bin",
        "..\\docs\\model_weight_bin\\encoder_embed_convnext_pointwise_conv2_bias.bin",
        "..\\docs\\model_weight_bin\\encoder_embed_out_weight.bin",
        "..\\docs\\model_weight_bin\\encoder_embed_out_bias.bin",
        "..\\docs\\model_weight_bin\\encoder_embed_out_norm_log_scale.bin",
        "..\\docs\\model_weight_bin\\encoder_embed_out_norm_bias.bin"
    };

    char *output_files[] = {
        "..\\docs\\inference_outputs_bin\\encoder_embed_conv_0.bin",
        "..\\docs\\inference_outputs_bin\\encoder_embed_conv_1.bin",
        "..\\docs\\inference_outputs_bin\\encoder_embed_conv_3.bin",
        "..\\docs\\inference_outputs_bin\\encoder_embed_conv_4.bin",
        "..\\docs\\inference_outputs_bin\\encoder_embed_conv_6.bin",
        "..\\docs\\inference_outputs_bin\\encoder_embed_conv_7.bin",
        "..\\docs\\inference_outputs_bin\\encoder_embed_conv_9.bin",
        "..\\docs\\inference_outputs_bin\\encoder_embed_convnext_depthwise_conv.bin",
        "..\\docs\\inference_outputs_bin\\encoder_embed_convnext_pointwise_conv1.bin",
        "..\\docs\\inference_outputs_bin\\encoder_embed_convnext_swooshL.bin",
        "..\\docs\\inference_outputs_bin\\encoder_embed_convnext_pointwise_conv2.bin",
        "..\\docs\\inference_outputs_bin\\encoder_embed_out.bin",
        "..\\docs\\inference_outputs_bin\\encoder_embed_out_norm.bin"
    };

    // Model input tensor
    Tensor *encoder_model_input; 

    // Conv_Embed conv2d weight and bias
    Tensor  *embed_conv_0_weight, 
            *embed_conv_4_weight, 
            *embed_conv_7_weight;

    Tensor  *embed_conv_0_bias, 
            *embed_conv_4_bias, 
            *embed_conv_7_bias;

    // Conv_Embed convnext weight and bias
    Tensor  *embed_convnext_depthwise_conv_weight, 
            *embed_convnext_depthwise_conv1_weight,
            *embed_convnext_depthwise_conv2_weight;

    Tensor  *embed_convnext_depthwise_conv_bias, 
            *embed_convnext_depthwise_conv1_bias,
            *embed_convnext_depthwise_conv2_bias;

    // Conv_Embed out weight and bias
    Tensor  *encoder_embed_out_weight, 
            *encoder_embed_out_bias;

    Tensor  *encoder_embed_out_norm_log_scale, 
            *encoder_embed_out_norm_bias;

    // Output tensors
    Tensor  *encoder_embed_conv_0, 
            *encoder_embed_conv_4, 
            *encoder_embed_conv_7;
    
    Tensor  *encoder_embed_convnext_depthwise_conv,
            *encoder_embed_convnext_pointwise_conv1,
            *encoder_embed_convnext_pointwise_conv2;

    Tensor  *encoder_embed_out;

    // Create input
    TENSOR_Create(&encoder_model_input, 1, 1, 313, 80);

    FILE_ReadTensorBinary(encoder_model_input, input_file);

    // Create parameters
    TENSOR_Create(&embed_conv_0_weight, 8, 1, 3, 3);
    TENSOR_Create(&embed_conv_0_bias, 8, 1, 1, 1);
    TENSOR_Create(&embed_conv_4_weight, 32, 8, 3, 3);
    TENSOR_Create(&embed_conv_4_bias, 32, 1, 1, 1);
    TENSOR_Create(&embed_conv_7_weight, 128, 32, 3, 3);
    TENSOR_Create(&embed_conv_7_bias, 128, 1, 1, 1);
        
    FILE_ReadTensorBinary(embed_conv_0_weight, param_files[0]);
    FILE_ReadTensorBinary(embed_conv_0_bias, param_files[1]);
    FILE_ReadTensorBinary(embed_conv_4_weight, param_files[2]);
    FILE_ReadTensorBinary(embed_conv_4_bias, param_files[3]);
    FILE_ReadTensorBinary(embed_conv_7_weight, param_files[4]);
    FILE_ReadTensorBinary(embed_conv_7_bias, param_files[5]);

    // Create output tensors
    TENSOR_Create(&encoder_embed_conv_0, 1, 8, 311, 80);
    TENSOR_Create(&encoder_embed_conv_4, 1, 32, 155, 39);
    TENSOR_Create(&encoder_embed_conv_7, 1, 128, 153, 19);

    // conv2d
    TENSOR_conv2d(encoder_model_input, encoder_embed_conv_0, embed_conv_0_weight, embed_conv_0_bias, 1, 1, 0, 1, 1);

    Check_Intermediate(encoder_embed_conv_0, output_files[0]);

    TENSOR_Swoosh(encoder_embed_conv_0, 'r');

    Check_Intermediate(encoder_embed_conv_0, output_files[2]);

    TENSOR_Free(&encoder_model_input); 
    TENSOR_Free(&embed_conv_0_weight);
    TENSOR_Free(&embed_conv_0_bias);

    TENSOR_conv2d(encoder_embed_conv_0, encoder_embed_conv_4, embed_conv_4_weight, embed_conv_4_bias, 2, 2, 0, 0, 1);

    Check_Intermediate(encoder_embed_conv_4, output_files[3]);

    TENSOR_Swoosh(encoder_embed_conv_4, 'r');

    Check_Intermediate(encoder_embed_conv_4, output_files[4]);

    TENSOR_Free(&encoder_embed_conv_0); 
    TENSOR_Free(&embed_conv_4_weight);
    TENSOR_Free(&embed_conv_4_bias);

    TENSOR_conv2d(encoder_embed_conv_4, encoder_embed_conv_7, embed_conv_7_weight, embed_conv_7_bias, 1, 2, 0, 0, 1);

    Check_Intermediate(encoder_embed_conv_7, output_files[5]);

    TENSOR_Swoosh(encoder_embed_conv_7, 'r');

    Check_Intermediate(encoder_embed_conv_7, output_files[6]);

    TENSOR_Free(&encoder_embed_conv_4);
    TENSOR_Free(&embed_conv_7_weight);
    TENSOR_Free(&embed_conv_7_bias);                


    // convnext
    TENSOR_Create(&embed_convnext_depthwise_conv_weight, 128, 1, 7, 7);
    TENSOR_Create(&embed_convnext_depthwise_conv_bias, 128, 1, 1, 1);
    TENSOR_Create(&embed_convnext_depthwise_conv1_weight, 384, 128, 1, 1);
    TENSOR_Create(&embed_convnext_depthwise_conv1_bias, 384, 1, 1, 1);
    TENSOR_Create(&embed_convnext_depthwise_conv2_weight, 128, 384, 1, 1);
    TENSOR_Create(&embed_convnext_depthwise_conv2_bias, 128, 1, 1, 1);

    TENSOR_Create(&encoder_embed_convnext_depthwise_conv, 1, 128, 153, 19);
    TENSOR_Create(&encoder_embed_convnext_pointwise_conv1, 1, 384, 153, 19);
    TENSOR_Create(&encoder_embed_convnext_pointwise_conv2, 1, 128, 153, 19);

    FILE_ReadTensorBinary(embed_convnext_depthwise_conv_weight, param_files[6]);
    FILE_ReadTensorBinary(embed_convnext_depthwise_conv_bias, param_files[7]);
    FILE_ReadTensorBinary(embed_convnext_depthwise_conv1_weight, param_files[8]);
    FILE_ReadTensorBinary(embed_convnext_depthwise_conv1_bias, param_files[9]);
    FILE_ReadTensorBinary(embed_convnext_depthwise_conv2_weight, param_files[10]);
    FILE_ReadTensorBinary(embed_convnext_depthwise_conv2_bias, param_files[11]);


    channel = encoder_embed_conv_7->dim2;

    TENSOR_conv2d(encoder_embed_conv_7, encoder_embed_convnext_depthwise_conv, embed_convnext_depthwise_conv_weight, embed_convnext_depthwise_conv_bias, 1, 1, 3, 3, channel);

    Check_Intermediate(encoder_embed_convnext_depthwise_conv, output_files[7]);

    TENSOR_Free(&embed_convnext_depthwise_conv_weight);
    TENSOR_Free(&embed_convnext_depthwise_conv_bias);

    TENSOR_conv2d(encoder_embed_convnext_depthwise_conv, encoder_embed_convnext_pointwise_conv1, embed_convnext_depthwise_conv1_weight, embed_convnext_depthwise_conv1_bias, 1, 1, 0, 0, 1); 

    Check_Intermediate(encoder_embed_convnext_pointwise_conv1, output_files[8]);
    
    TENSOR_Swoosh(encoder_embed_convnext_pointwise_conv1, 'l');

    Check_Intermediate(encoder_embed_convnext_pointwise_conv1, output_files[9]);

    TENSOR_Free(&encoder_embed_convnext_depthwise_conv);
    TENSOR_Free(&embed_convnext_depthwise_conv1_weight);
    TENSOR_Free(&embed_convnext_depthwise_conv1_bias); 

    TENSOR_conv2d(encoder_embed_convnext_pointwise_conv1, encoder_embed_convnext_pointwise_conv2, embed_convnext_depthwise_conv2_weight, embed_convnext_depthwise_conv2_bias, 1, 1, 0, 0, 1);

    Check_Intermediate(encoder_embed_convnext_pointwise_conv2, output_files[10]);

    TENSOR_Free(&encoder_embed_convnext_pointwise_conv1);
    TENSOR_Free(&embed_convnext_depthwise_conv2_weight);
    TENSOR_Free(&embed_convnext_depthwise_conv2_bias);

    TENSOR_Add(encoder_embed_convnext_pointwise_conv2, encoder_embed_conv_7); //Bypass

    int perm[4] = {0, 2, 1, 3};

    TENSOR_Transpose(encoder_embed_convnext_pointwise_conv2, perm, 4);

    TENSOR_Reshape(encoder_embed_convnext_pointwise_conv2, 1, 1, encoder_embed_convnext_pointwise_conv2->dim2, encoder_embed_convnext_pointwise_conv2->dim3 * encoder_embed_convnext_pointwise_conv2->dim4); 

    TENSOR_Free(&encoder_embed_conv_7);

    // linear
    TENSOR_Create(&encoder_embed_out_weight, 1, 1, 192, 2432);
    TENSOR_Create(&encoder_embed_out_bias, 1, 1, 1, 192);
    
    TENSOR_Create(&encoder_embed_out, 1, 1, 153, 192);
        
    FILE_ReadTensorBinary(encoder_embed_out_weight, param_files[12]);
    FILE_ReadTensorBinary(encoder_embed_out_bias, param_files[13]);


    TENSOR_Linear(encoder_embed_convnext_pointwise_conv2, encoder_embed_out, encoder_embed_out_weight, encoder_embed_out_bias);

    Check_Intermediate(encoder_embed_out, output_files[11]);

    TENSOR_Free(&encoder_embed_convnext_pointwise_conv2);
    TENSOR_Free(&encoder_embed_out_weight);
    TENSOR_Free(&encoder_embed_out_bias);

    // Bias Norm
    TENSOR_Create(&encoder_embed_out_norm_bias, 1, 1, 1, 192);

    FILE_ReadTensorBinary(encoder_embed_out_norm_bias, param_files[15]);

    log_scale = read_parameters("D:\\zip_C\\docs\\model_weight_bin\\encoder_embed_out_norm_log_scale.bin");

    biasnorm(encoder_embed_out, encoder_embed_out_norm_bias, log_scale);

    TENSOR_Free(&encoder_embed_out_norm_bias);

    Check_Intermediate(encoder_embed_out, output_files[12]);
    return 0;
}