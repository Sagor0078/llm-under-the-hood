import torch

# Q, K, V, output are tensors on the GPU
def solve(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, output: torch.Tensor,
          M: int, N: int, d: int, alpha: float):

    # compute raw attention scores (M x N)
    scores = torch.matmul(Q, K.transpose(0, 1)) / (d ** 0.5)

    # relative position bias Δ = i - j
    i_idx = torch.arange(M, device=Q.device).unsqueeze(1)  # (M,1)
    j_idx = torch.arange(N, device=Q.device).unsqueeze(0)  # (1,N)
    delta = i_idx - j_idx   # (M, N)

    # add ALiBi bias
    scores = scores + alpha * delta

    # softmax row-wise
    attn_weights = torch.softmax(scores, dim=-1)

    # multiply by V
    output[:] = torch.matmul(attn_weights, V)
