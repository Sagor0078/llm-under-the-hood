import torch

def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    input = input.to(torch.float32)

    half = N // 2
    x1 = input[:half]
    x2 = input[half:]

    sigmoid_x1 = 1.0 / (1.0 + torch.exp(-x1))

    # compute SiLU(x1) = x1 * sigmoid(x1)
    silu_x1 = x1 * sigmoid_x1

    # compute SWiGLU = silu(x1) * x2
    result = silu_x1 * x2

    # store result into output
    output[:half].copy_(result)
