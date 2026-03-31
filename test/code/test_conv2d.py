import torch
import torch.nn.functional as F
import ctypes
import numpy as np
import os

# --- CẤU HÌNH ĐƯỜNG DẪN ---
# Đảm bảo file tensor.dll nằm cùng thư mục với file này
DLL_PATH = os.path.abspath("tensor.dll")

# --- ĐỊNH NGHĨA CTYPES ---
class C_Tensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.POINTER(ctypes.c_double)),
        ("dim1", ctypes.c_int),
        ("dim2", ctypes.c_int),
        ("dim3", ctypes.c_int),
        ("dim4", ctypes.c_int),
    ]

# Nạp thư viện
try:
    c_lib = ctypes.CDLL(DLL_PATH)
except OSError as e:
    print(f"Không thể nạp DLL: {e}")
    exit()

# Khai báo kiểu dữ liệu cho hàm C
c_lib.TENSOR_Create.argtypes = [ctypes.POINTER(ctypes.POINTER(C_Tensor)), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
c_lib.TENSOR_Init.argtypes = [ctypes.POINTER(ctypes.POINTER(C_Tensor))]
c_lib.TENSOR_conv2d.argtypes = [
    ctypes.POINTER(C_Tensor), # imf
    ctypes.POINTER(C_Tensor), # omf
    ctypes.POINTER(C_Tensor), # kernel
    ctypes.POINTER(C_Tensor), # bias
    ctypes.c_int,             # stride
    ctypes.c_int,             # padding
    ctypes.c_int              # groups
]

def to_c_tensor(torch_tensor):
    """Chuyển đổi Torch Tensor (N,C,H,W) sang C Tensor Struct"""
    shape = torch_tensor.shape
    # Giả định thứ tự trong struct C: dim1=N, dim2=C, dim3=H, dim4=W
    d1, d2, d3, d4 = shape[0], shape[1], shape[2], shape[3]
    
    c_tensor_ptr = ctypes.POINTER(C_Tensor)()
    c_lib.TENSOR_Create(ctypes.byref(c_tensor_ptr), d1, d2, d3, d4)
    
    # Copy dữ liệu (Torch float32 -> C double)
    data_np = torch_tensor.detach().numpy().astype(np.float64).flatten()
    for i in range(len(data_np)):
        c_tensor_ptr.contents.data[i] = data_np[i]
    return c_tensor_ptr

def test_conv2d_complete():
    # --- 1. Thông số thử nghiệm ---
    N, C_in, C_out = 1, 4, 4
    H_in, W_in = 5, 5
    K_h, K_w = 3, 3
    stride = 1
    padding = 1
    groups = 2 # Test thử Grouped Convolution

    # --- 2. Tạo dữ liệu bằng PyTorch ---
    input_pt = torch.randn(N, C_in, H_in, W_in)
    weight_pt = torch.randn(C_out, C_in // groups, K_h, K_w)
    bias_pt = torch.randn(C_out)

    # Tính toán kết quả mong đợi từ PyTorch
    output_pt = F.conv2d(input_pt, weight_pt, bias_pt, stride=stride, padding=padding, groups=groups)

    # --- 3. Chuyển đổi sang C Tensor ---
    c_imf = to_c_tensor(input_pt)
    c_kernel = to_c_tensor(weight_pt)
    
    # Bias trong C của bạn là 1D data, nhưng struct yêu cầu 4 dims. 
    # Ta tạo tensor (C_out, 1, 1, 1)
    c_bias = to_c_tensor(bias_pt.view(C_out, 1, 1, 1))

    # Tạo Output Tensor trong C
    H_out, W_out = output_pt.shape[2], output_pt.shape[3]
    c_omf = ctypes.POINTER(C_Tensor)()
    c_lib.TENSOR_Create(ctypes.byref(c_omf), N, C_out, H_out, W_out)
    c_lib.TENSOR_Init(ctypes.byref(c_omf))

    # --- 4. Gọi hàm C ---
    print("Đang chạy TENSOR_conv2d...")
    result = c_lib.TENSOR_conv2d(c_imf, c_omf, c_kernel, c_bias, stride, padding, groups)

    # --- 5. So sánh ---
    # Lấy dữ liệu từ C về numpy
    c_output_raw = np.array([c_omf.contents.data[i] for i in range(N*C_out*H_out*W_out)])
    c_output_shaped = c_output_raw.reshape(output_pt.shape)

    max_diff = np.max(np.abs(c_output_shaped - output_pt.detach().numpy()))
    
    print("-" * 30)
    print(f"Kết quả Max Difference: {max_diff:.10e}")
    if max_diff < 1e-7:
        print("✅ THÀNH CÔNG: Kết quả khớp với PyTorch!")
    else:
        print("❌ THẤT BẠI: Kết quả sai lệch quá lớn.")
    print("-" * 30)

if __name__ == "__main__":
    test_conv2d_complete()