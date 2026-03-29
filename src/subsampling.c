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

int main()
{
    int channel, feature_dim, in_channels;
    double log_scale;

    char *input_file = "..\\docs\\input_features_bin\\input_features.bin"; 

    char *param_files[] = {
        "..\\docs\\encoder_embed_conv_0_weight.bin",
        "..\\docs\\encoder_embed_conv_0_bias.bin",
        "..\\docs\\encoder_embed_conv_4_weight.bin",
        "..\\docs\\encoder_embed_conv_4_bias.bin",
        "..\\docs\\encoder_embed_conv_7_weight.bin",
        "..\\docs\\encoder_embed_conv_7_bias.bin",
        "..\\docs\\encoder_embed_convnext_depthwise_conv_weight.bin",
        "..\\docs\\encoder_embed_convnext_depthwise_conv_bias.bin",
        "..\\docs\\encoder_embed_convnext_pointwise_conv1_weight.bin",
        "..\\docs\\encoder_embed_convnext_pointwise_conv1_bias.bin",
        "..\\docs\\encoder_embed_convnext_pointwise_conv2_weight.bin",
        "..\\docs\\encoder_embed_convnext_pointwise_conv2_bias.bin",
        "..\\docs\\encoder_embed_out_weight.bin",
        "..\\docs\\encoder_embed_out_bias.bin",
        "..\\docs\\encoder_embed_out_norm_log_scale.bin",
        "..\\docs\\encoder_embed_out_norm_bias.bin"
    };

//     char *output_files[] = {
//         "..\\docs\\encoder_embed_conv_0.bin",
//         "..\\docs\\encoder_embed_conv_1.bin",
//         "..\\docs\\encoder_embed_conv_3.bin",
//         "..\\docs\\encoder_embed_conv_4.bin",
//         "..\\docs\\encoder_embed_conv_6.bin",
//         "..\\docs\\encoder_embed_conv_7.bin",
//         "..\\docs\\encoder_embed_conv_9.bin",
//         "..\\docs\\encoder_embed_convnext_depthwise_conv.bin",
//         "..\\docs\\encoder_embed_convnext_pointwise_conv1.bin",
//         "..\\docs\\encoder_embed_convnext_swooshL.bin",
//         "..\\docs\\encoder_embed_convnext_pointwise_conv2.bin",
//         "..\\docs\\encoder_embed_out.bin",
//         "..\\docs\\encoder_embed_out_norm.bin"
//     };

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

    FILE_ReadTensorBinary(input_file, encoder_model_input);

    // Create parameters
    TENSOR_Create(&embed_conv_0_weight, 8, 1, 3, 3);
    TENSOR_Create(&embed_conv_0_bias, 8, 1, 1, 1);
    TENSOR_Create(&embed_conv_4_weight, 32, 8, 3, 3);
    TENSOR_Create(&embed_conv_4_bias, 32, 1, 1, 1);
    TENSOR_Create(&embed_conv_7_weight, 128, 32, 3, 3);
    TENSOR_Create(&embed_conv_7_bias, 128, 1, 1, 1);
        
    FILE_ReadTensorBinary(param_files[0], embed_conv_0_weight);
    FILE_ReadTensorBinary(param_files[1], embed_conv_0_bias);
    FILE_ReadTensorBinary(param_files[2], embed_conv_4_weight);
    FILE_ReadTensorBinary(param_files[3], embed_conv_4_bias);
    FILE_ReadTensorBinary(param_files[4], embed_conv_7_weight);
    FILE_ReadTensorBinary(param_files[5], embed_conv_7_bias);

    // Create output tensors
    TENSOR_Create(&encoder_embed_conv_0, 1, 8, 311, 80);
    TENSOR_Create(&encoder_embed_conv_4, 1, 32, 155, 39);
    TENSOR_Create(&encoder_embed_conv_7, 1, 128, 153, 19);

    // conv2d
    TENSOR_conv2d(encoder_model_input, encoder_embed_conv_0, embed_conv_0_weight, embed_conv_0_bias, 1, 1, 0, 1, 1);
    TENSOR_Swoosh(encoder_embed_conv_0, 'r');

    TENSOR_Free(&encoder_model_input); 
    TENSOR_Free(&embed_conv_0_weight);
    TENSOR_Free(&embed_conv_0_bias);

    TENSOR_conv2d(encoder_embed_conv_0, encoder_embed_conv_4, embed_conv_4_weight, embed_conv_4_bias, 2, 2, 0, 0, 1);
    TENSOR_Swoosh(encoder_embed_conv_4, 'r');

    TENSOR_Free(&encoder_embed_conv_0); 
    TENSOR_Free(&embed_conv_4_weight);
    TENSOR_Free(&embed_conv_4_bias);

    TENSOR_conv2d(encoder_embed_conv_4, encoder_embed_conv_7, embed_conv_7_weight, embed_conv_7_bias, 1, 2, 0, 0, 1);
    TENSOR_Swoosh(encoder_embed_conv_7, 'r');

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

    FILE_ReadTensorBinary(param_files[6], embed_convnext_depthwise_conv_weight);
    FILE_ReadTensorBinary(param_files[7], embed_convnext_depthwise_conv_bias);
    FILE_ReadTensorBinary(param_files[8], embed_convnext_depthwise_conv1_weight);
    FILE_ReadTensorBinary(param_files[9], embed_convnext_depthwise_conv1_bias);
    FILE_ReadTensorBinary(param_files[10], embed_convnext_depthwise_conv2_weight);
    FILE_ReadTensorBinary(param_files[11], embed_convnext_depthwise_conv2_bias);


    channel = encoder_embed_conv_7->dim2;

    TENSOR_conv2d(encoder_embed_conv_7, encoder_embed_convnext_depthwise_conv, embed_convnext_depthwise_conv_weight, embed_convnext_depthwise_conv_bias, 1, 1, 3, 3, channel);

    TENSOR_Free(&embed_convnext_depthwise_conv_weight);
    TENSOR_Free(&embed_convnext_depthwise_conv_bias);

    TENSOR_conv2d(encoder_embed_convnext_depthwise_conv, encoder_embed_convnext_pointwise_conv1, embed_convnext_depthwise_conv1_weight, embed_convnext_depthwise_conv1_bias, 1, 1, 0, 0, 1); 
    
    TENSOR_Swoosh(encoder_embed_convnext_pointwise_conv1, 'l');

    TENSOR_Free(&encoder_embed_convnext_depthwise_conv);
    TENSOR_Free(&embed_convnext_depthwise_conv1_weight);
    TENSOR_Free(&embed_convnext_depthwise_conv1_bias); 

    TENSOR_conv2d(encoder_embed_convnext_pointwise_conv1, encoder_embed_convnext_pointwise_conv2, embed_convnext_depthwise_conv2_weight, embed_convnext_depthwise_conv2_bias, 1, 1, 0, 0, 1);

    TENSOR_Free(&encoder_embed_convnext_pointwise_conv1);
    TENSOR_Free(&embed_convnext_depthwise_conv2_weight);
    TENSOR_Free(&embed_convnext_depthwise_conv2_bias);

    TENSOR_Add(encoder_embed_convnext_pointwise_conv2, encoder_embed_conv_7); //Bypass

    int new_dim[4] = {0};
    new_dim[0] = encoder_embed_conv_7->dim1;
    new_dim[1] = encoder_embed_conv_7->dim3;
    new_dim[2] = encoder_embed_conv_7->dim2;
    new_dim[3] = encoder_embed_conv_7->dim4;

    TENSOR_Transpose(encoder_embed_convnext_pointwise_conv2, &new_dim, 4);

    TENSOR_Reshape(encoder_embed_convnext_pointwise_conv2, 1, encoder_embed_convnext_pointwise_conv2->dim1, encoder_embed_convnext_pointwise_conv2->dim3, encoder_embed_convnext_pointwise_conv2->dim2 * encoder_embed_convnext_pointwise_conv2->dim4);

    TENSOR_Free(&encoder_embed_conv_7);

    // linear
    TENSOR_Create(&encoder_embed_out_weight, 192, 2432, 1, 1);
    TENSOR_Create(&encoder_embed_out_bias, 192, 1, 1, 1);
    
    TENSOR_Create(&encoder_embed_out, 1, 153, 192, 1);
        
    FILE_ReadTensorBinary(param_files[12], encoder_embed_out_weight);
    FILE_ReadTensorBinary(param_files[13], encoder_embed_out_bias);


    TENSOR_Linear(encoder_embed_convnext_pointwise_conv2, encoder_embed_out, encoder_embed_out_weight, encoder_embed_out_bias);

    TENSOR_Free(&encoder_embed_convnext_pointwise_conv2);
    TENSOR_Free(&encoder_embed_out_weight);
    TENSOR_Free(&encoder_embed_out_bias);

    // Bias Norm
    TENSOR_Create(&encoder_embed_out_norm_bias, 192, 1, 1, 1);

    FILE_ReadTensorBinary(param_files[15], encoder_embed_out_norm_bias);

    log_scale = read_parameters("D:\\AI_C\\parameters\\encoder_embed_out_norm_log_scale.bin");

    biasnorm(encoder_embed_out, encoder_embed_out_norm_bias, log_scale);

    TENSOR_Free(&encoder_embed_out_norm_bias);
}