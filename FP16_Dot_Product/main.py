import torch

# A, B, result are tensors on the GPU
def solve(A: torch.Tensor, B: torch.Tensor, result: torch.Tensor, N: int):

    if N == 0:
        with torch.no_grad():
            result.copy_(torch.tensor(0, dtype=torch.float16, device=result.device))
        return

    # ensure we don't track gradients and operate on device
    with torch.no_grad():

        # convert to float32 for multiplication + accumulation
        # use .float() (same device) then dot / sum in float32
        acc = (A[:N].float() * B[:N].float()).sum(dtype=torch.float32)

        # convert accumulated FP32 result to FP16 and write into result tensor
        acc_half = acc.to(torch.float16)

        result.copy_(acc_half)
