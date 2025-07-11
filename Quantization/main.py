# Quantized Matrix Multiplication (INT8) Solution on LeetGPU

import torch

def solve(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, M: int, N: int, K: int, scale_A: float, scale_B: float, scale_C: float, zero_point_A: int, zero_point_B: int, zero_point_C: int):
    A_shifted = A.to(torch.float32) - zero_point_A
    B_shifted = B.to(torch.float32) - zero_point_B

    accumulated_sum_tensor = torch.matmul(A_shifted, B_shifted)

    intermediate_scaled_product = accumulated_sum_tensor * scale_A * scale_B
    scaled_value_tensor = intermediate_scaled_product / scale_C

    final_float_value_tensor = scaled_value_tensor + zero_point_C

    rounded_tensor = torch.round(final_float_value_tensor)

    clamped_tensor = torch.clamp(rounded_tensor, -128, 127)

    C.copy_(clamped_tensor.to(torch.int8))
