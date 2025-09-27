import torch

# Q, K, V, output are tensors on the GPU
def solve(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, output: torch.Tensor,
          M: int, d: int):

    # define φ(x) = ELU(x) + 1
    phi = torch.nn.functional.elu

    def feature_map(x):
        return phi(x) + 1.0

    Q_phi = feature_map(Q)   # (M, d)
    K_phi = feature_map(K)   # (M, d)

    # compute numerator: Qφ (Kφ^T V)
    KV = torch.matmul(K_phi.T, V)          
    numerator = torch.matmul(Q_phi, KV)    

    # compute denominator: Qφ (sum_j Kφ_j)
    K_sum = torch.sum(K_phi, dim=0)        # (d,)
    denominator = torch.matmul(Q_phi, K_sum)  # (M,)

    # normalize: divide each row of numerator by corresponding denominator
    output[:] = numerator / denominator.unsqueeze(-1)
