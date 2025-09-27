import torch

# Q, K, V, output are tensors on the GPU
def solve(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, output: torch.Tensor,
          M: int, d: int):

    # compute attention scores (M x M)
    scores = torch.matmul(Q, K.transpose(0, 1)) / (d ** 0.5)

    # apply causal mask (upper-triangular part -> -inf)
    mask = torch.triu(torch.ones(M, M, device=Q.device), diagonal=1)
    scores = scores.masked_fill(mask.bool(), float('-inf'))

    # softmax row-wise
    attn_weights = torch.softmax(scores, dim=-1)

    # multiply by V
    output[:] = torch.matmul(attn_weights, V)
