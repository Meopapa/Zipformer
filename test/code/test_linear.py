import torch
import torch.nn as nn
import ctypes
import numpy as np
import os

# --- CẤU HÌNH ---
DLL_PATH = os.path.abspath("tensor.dll")

class C_Tensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.POINTER(ctypes.c_double)),
        ("dim1", ctypes.c_int),
        ("dim2", ctypes.c_int),
        ("dim3", ctypes.c_int),
        ("dim4", ctypes.c_int),
    ]

# Nạp thư viện
c_lib = ctypes.CDLL(DLL_PATH)

# Khai báo kiểu hàm
c_lib.TENSOR_Create.argtypes = [ctypes.POINTER(ctypes.POINTER(C_Tensor)), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
c_lib.TENSOR_Init.argtypes = [ctypes.POINTER(ctypes.POINTER(C_Tensor))]
c_lib.TENSOR_Linear.argtypes = [
    ctypes.POINTER(C_Tensor), # imf
    ctypes.POINTER(C_Tensor), # omf
    ctypes.POINTER(C_Tensor), # weight
    ctypes.POINTER(C_Tensor)  # bias
]

def to_c_tensor(torch_tensor, dims=None):
    """Chuyển Torch Tensor sang C struct. dims=[d1, d2, d3, d4]"""
    if dims is None:
        # Nếu là tensor 2D [Batch, Features], ta giả định d1=Batch, d2=1, d3=1, d4=Features
        s = torch_tensor.shape
        if len(s) == 2: dims = [s[0], 1, 1, s[1]]
        elif len(s) == 4: dims = [s[0], s[1], s[2], s[3]]
        else: dims = [1, 1, 1, s[0]]

    c_ptr = ctypes.POINTER(C_Tensor)()
    c_lib.TENSOR_Create(ctypes.byref(c_ptr), dims[0], dims[1], dims[2], dims[3])
    
    data_np = torch_tensor.detach().numpy().astype(np.float64).flatten()
    for i in range(len(data_np)):
        c_ptr.contents.data[i] = data_np[i]
    return c_ptr

def test_linear():
    # --- 1. Thông số (BatchSize=2, Seq=3, In=16, Out=32) ---
    # Trong Zipformer, Linear thường nhận đầu vào 3D hoặc 4D
    N, D2, D3, In_F = 2, 1, 3, 16 
    Out_F = 32

    # --- 2. Tạo dữ liệu PyTorch ---
    input_pt = torch.randn(N, D2, D3, In_F)
    # Trọng số Linear trong Torch là [Out, In]
    weight_pt = torch.randn(Out_F, In_F)
    bias_pt = torch.randn(Out_F)

    # Kết quả mong đợi
    # nn.Linear chỉ tác động lên chiều cuối cùng
    linear_layer = nn.Linear(In_F, Out_F)
    linear_layer.weight.data = weight_pt
    linear_layer.bias.data = bias_pt
    output_pt = linear_layer(input_pt)

    # --- 3. Chuyển sang C ---
    c_imf = to_c_tensor(input_pt, [N, D2, D3, In_F])
    # Weight trong C của bạn dùng TENSOR_Index(0,0,o_f, i_f) -> [1, 1, Out, In]
    c_weight = to_c_tensor(weight_pt, [1, 1, Out_F, In_F])
    # Bias trong C: ta giả định dim4 là chiều chứa data [1, 1, 1, Out]
    c_bias = to_c_tensor(bias_pt, [1, 1, 1, Out_F])

    # Tạo Output
    c_omf = ctypes.POINTER(C_Tensor)()
    c_lib.TENSOR_Create(ctypes.byref(c_omf), N, D2, D3, Out_F)
    c_lib.TENSOR_Init(ctypes.byref(c_omf))

    # --- 4. Chạy hàm C ---
    print("Đang chạy TENSOR_Linear...")
    c_lib.TENSOR_Linear(c_imf, c_omf, c_weight, c_bias)

    # --- 5. So sánh ---
    c_out_raw = np.array([c_omf.contents.data[i] for i in range(N*D2*D3*Out_F)])
    c_out_shaped = c_out_raw.reshape(output_pt.shape)

    max_diff = np.max(np.abs(c_out_shaped - output_pt.detach().numpy()))
    
    print("-" * 30)
    print(f"Max Difference: {max_diff:.10e}")
    
    # Sai số cho phép đối với phép nhân ma trận lớn
    if max_diff < 1e-6:
        print("✅ THÀNH CÔNG: Linear khớp với PyTorch!")
    else:
        print("❌ THẤT BẠI: Kết quả sai lệch.")
    print("-" * 30)

if __name__ == "__main__":
    test_linear()