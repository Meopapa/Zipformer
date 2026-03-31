import ctypes
import os
import numpy as np

# --- 1. Load thư viện DLL ---
dll_path = os.path.abspath("D:\\AI_C\\test\\code\\libloss.dll")
if not os.path.exists(dll_path):
    raise FileNotFoundError(f"Không tìm thấy file: {dll_path}")

lib = ctypes.CDLL(dll_path)

# --- 2. Khai báo Prototype cho các hàm C ---
# Tất cả các hàm đều có signature: double (double*, double*, int)
functions = [lib.LOSS_mse, lib.LOSS_mae, lib.LOSS_rmse, lib.LOSS_cross_entropy]

for func in functions:
    func.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_int]
    func.restype = ctypes.c_double

# --- 3. Hàm hỗ trợ so sánh ---
def verify(name, c_val, py_val):
    status = "✅ PASS" if np.isclose(c_val, py_val, atol=1e-7) else "❌ FAIL"
    print(f"[{name}]")
    print(f"  C:      {c_val:.10f}")
    print(f"  Numpy:  {py_val:.10f}")
    print(f"  Result: {status}\n")

# --- 4. Thực hiện Test ---

# A. Test Regression (MSE, MAE, RMSE)
y_true_reg = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
y_pred_reg = np.array([1.1, 1.9, 3.2, 3.8], dtype=np.float64)
n_reg = len(y_true_reg)

ptr_true_reg = y_true_reg.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
ptr_pred_reg = y_pred_reg.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

# Tính bằng Numpy
mse_np = np.mean((y_true_reg - y_pred_reg)**2)
mae_np = np.mean(np.abs(y_true_reg - y_pred_reg))
rmse_np = np.sqrt(mse_np)

verify("MSE", lib.LOSS_mse(ptr_true_reg, ptr_pred_reg, n_reg), mse_np)
verify("MAE", lib.LOSS_mae(ptr_true_reg, ptr_pred_reg, n_reg), mae_np)
verify("RMSE", lib.LOSS_rmse(ptr_true_reg, ptr_pred_reg, n_reg), rmse_np)

# B. Test Binary Cross Entropy (BCE) - Khi n=1 (hoặc theo logic n <= 1 của bạn)
y_true_bce = np.array([1.0], dtype=np.float64)
y_pred_bce = np.array([0.8], dtype=np.float64)
eps = 1e-15
bce_np = -(y_true_bce * np.log(y_pred_bce) + (1 - y_true_bce) * np.log(1 - y_pred_bce))

verify("Binary Cross Entropy", 
       lib.LOSS_cross_entropy(y_true_bce.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 
                              y_pred_bce.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 1), 
       bce_np[0])

# C. Test Categorical Cross Entropy (CCE) - Khi n > 1
y_true_cce = np.array([0, 1, 0], dtype=np.float64) # Giả sử lớp đúng là index 1
y_pred_cce = np.array([0.2, 0.7, 0.1], dtype=np.float64)
cce_np = -np.sum(y_true_cce * np.log(y_pred_cce))

verify("Categorical Cross Entropy", 
       lib.LOSS_cross_entropy(y_true_cce.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 
                              y_pred_cce.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 3), 
       cce_np)