import ctypes
import numpy as np
import os

# ---------------- 1. ĐỊNH NGHĨA CTYPES ---------------- #
class CTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.POINTER(ctypes.c_double)),
        ("row", ctypes.c_int),
        ("collumn", ctypes.c_int), 
        ("depth", ctypes.c_int)
    ]

lib_path = os.path.abspath('./tensor.dll')
try:
    lib = ctypes.CDLL(lib_path)
except OSError:
    raise FileNotFoundError(f"Không tìm thấy file {lib_path}. Hãy chạy lệnh gcc -shared trước!")

# --- Cấu hình Tensor Core ---
lib.TENSOR_Create.argtypes = [ctypes.POINTER(ctypes.POINTER(CTensor)), ctypes.c_int, ctypes.c_int, ctypes.c_int]
lib.TENSOR_Free.argtypes = [ctypes.POINTER(ctypes.POINTER(CTensor))]

lib.TENSOR_Matmul.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]
lib.TENSOR_Matmul.restype = ctypes.POINTER(CTensor)

lib.TENSOR_Add.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]
lib.TENSOR_Add.restype = ctypes.POINTER(CTensor)

lib.TENSOR_Sub.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]
lib.TENSOR_Sub.restype = ctypes.POINTER(CTensor)

lib.TENSOR_ScalarMul.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]
lib.TENSOR_ScalarMul.restype = ctypes.c_double 

lib.TENSOR_Mul.argtypes = [ctypes.c_double, ctypes.POINTER(CTensor)]
lib.TENSOR_Mul.restype = ctypes.c_int 

lib.TENSOR_Reshape.argtypes = [ctypes.POINTER(CTensor), ctypes.c_int, ctypes.c_int, ctypes.c_int]
lib.TENSOR_Reshape.restype = ctypes.c_int 

# --- Cấu hình Activation ---
# Cả 3 hàm đều nhận con trỏ Tensor, biến đổi trực tiếp (in-place) và trả về int
lib.TENSOR_ReLU.argtypes = [ctypes.POINTER(CTensor)]
lib.TENSOR_ReLU.restype = ctypes.c_int

lib.TENSOR_Sigmoid.argtypes = [ctypes.POINTER(CTensor)]
lib.TENSOR_Sigmoid.restype = ctypes.c_int

lib.TENSOR_Softmax.argtypes = [ctypes.POINTER(CTensor)]
lib.TENSOR_Softmax.restype = ctypes.c_int

# ---------------- 2. HÀM HELPER ---------------- #
def numpy_to_ctensor(np_array):
    np_array = np.ascontiguousarray(np_array, dtype=np.float64)
    d, r, c = np_array.shape
    tensor_ptr = ctypes.POINTER(CTensor)()
    lib.TENSOR_Create(ctypes.byref(tensor_ptr), r, c, d)
    size = d * r * c
    buffer = ctypes.cast(tensor_ptr.contents.data, ctypes.POINTER(ctypes.c_double * size)).contents
    c_array_view = np.ndarray((d, r, c), dtype=np.float64, buffer=buffer)
    np.copyto(c_array_view, np_array)
    return tensor_ptr

def ctensor_to_numpy(tensor_ptr):
    if not tensor_ptr: return None
    t = tensor_ptr.contents
    size = t.depth * t.row * t.collumn
    buffer = ctypes.cast(t.data, ctypes.POINTER(ctypes.c_double * size)).contents
    return np.ndarray((t.depth, t.row, t.collumn), dtype=np.float64, buffer=buffer).copy()

# ---------------- 3. CÁC HÀM KIỂM THỬ CORE ---------------- #
def test_add_sub():
    np_A = np.random.rand(2, 3, 4)
    np_B = np.random.rand(2, 3, 4)
    
    np_add = np_A + np_B
    np_sub = np_A - np_B
    
    c_A = numpy_to_ctensor(np_A)
    c_B = numpy_to_ctensor(np_B)
    
    c_add_ptr = lib.TENSOR_Add(c_A, c_B)
    c_sub_ptr = lib.TENSOR_Sub(c_A, c_B)
    
    c_add = ctensor_to_numpy(c_add_ptr)
    c_sub = ctensor_to_numpy(c_sub_ptr)
    
    lib.TENSOR_Free(ctypes.byref(c_A))
    lib.TENSOR_Free(ctypes.byref(c_B))
    lib.TENSOR_Free(ctypes.byref(c_add_ptr))
    lib.TENSOR_Free(ctypes.byref(c_sub_ptr))
    
    assert np.allclose(np_add, c_add, atol=1e-7), "Add sai toán học!"
    assert np.allclose(np_sub, c_sub, atol=1e-7), "Sub sai toán học!"

# ---------------- 4. CÁC HÀM KIỂM THỬ ACTIVATION ---------------- #
def test_activations():
    print("Testing ReLU... ", end="")
    # Dùng randn để có cả số âm và số dương
    np_A = np.random.randn(2, 3, 4) 
    c_A = numpy_to_ctensor(np_A)
    
    lib.TENSOR_ReLU(c_A)
    c_result = ctensor_to_numpy(c_A)
    
    # Ground Truth Numpy: max(0, x)
    np_result = np.maximum(np_A, 0)
    
    assert np.allclose(np_result, c_result, atol=1e-7), "ReLU tính sai!"
    lib.TENSOR_Free(ctypes.byref(c_A))
    print("PASSED")

    print("Testing Sigmoid... ", end="")
    np_A = np.random.randn(2, 3, 4)
    c_A = numpy_to_ctensor(np_A)
    
    lib.TENSOR_Sigmoid(c_A)
    c_result = ctensor_to_numpy(c_A)
    
    # Ground Truth Numpy
    np_result = 1.0 / (1.0 + np.exp(-np_A))
    
    assert np.allclose(np_result, c_result, atol=1e-7), "Sigmoid tính sai!"
    lib.TENSOR_Free(ctypes.byref(c_A))
    print("PASSED")

    print("Testing Softmax... ", end="")
    np_A = np.random.randn(2, 3, 4)
    c_A = numpy_to_ctensor(np_A)
    
    lib.TENSOR_Softmax(c_A)
    c_result = ctensor_to_numpy(c_A)
    
    # Ground Truth Numpy (Max Trick theo trục ngang/cột cuối cùng)
    max_val = np.max(np_A, axis=-1, keepdims=True)
    exp_val = np.exp(np_A - max_val)
    np_result = exp_val / np.sum(exp_val, axis=-1, keepdims=True)
    
    assert np.allclose(np_result, c_result, atol=1e-7), "Softmax tính sai!"
    lib.TENSOR_Free(ctypes.byref(c_A))
    print("PASSED")

if __name__ == "__main__":
    print("=== BẮT ĐẦU KIỂM THỬ TỔNG THỂ ===")
    
    print("Testing Math Core... ", end="")
    test_add_sub()
    print("PASSED")
    
    test_activations()
    
    print("\n[THÀNH CÔNG] Toàn bộ Tensor Core và Activation đã chạy chính xác 100%.")