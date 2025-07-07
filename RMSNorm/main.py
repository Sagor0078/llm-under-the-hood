import torch
import torch.nn as nn
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RMSNorm(nn.Module):
    def __init__(self, dim: int, p: float = -1.0, eps: float = 1e-8, bias: bool = False):
        super().__init__() # Use super().__init__() for modern Python

        if not isinstance(dim, int) or dim <= 0:
            raise ValueError(f"dim must be a positive integer, but got {dim}")
        # Adjust warning for p outside [0, 1] but allowing -1.0
        if not (-1.0 <= p <= 1.0):
             logger.warning(f"Partial RMSNorm parameter 'p' is outside the valid range [-1.0, 1.0]. "
                           f"Full RMSNorm (p=1.0) will be applied. Current p: {p}")
        if eps <= 0:
            raise ValueError(f"epsilon 'eps' must be positive, but got {eps}")

        self.eps = eps
        self.dim = dim
        self.p = p
        self.bias = bias

        # scale parameter, initialized to ones
        self.scale = nn.Parameter(torch.ones(dim))

        # offset parameter, only if bias is True
        if self.bias:
            self.offset = nn.Parameter(torch.zeros(dim))
        else:
            # Register offset as None if bias is False
            self.register_parameter('offset', None)


    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if x.shape[-1] != self.dim:
            raise ValueError(f"Input tensor last dimension ({x.shape[-1]}) does not match "
                             f"RMSNorm's 'dim' ({self.dim}).")

        # Determine the normalization scope (full or partial)
        partial_size = self.dim # Default to full size

        if 0.0 <= self.p <= 1.0: # Partial RMSNorm is enabled
            partial_size = int(self.dim * self.p)

            if partial_size == 0 and self.p > 0 and self.dim > 0:
                 logger.warning(f"Partial size calculated as 0 for p={self.p} and dim={self.dim}. "
                                f"Consider adjusting p or dim. Normalization will proceed with a minimal non-zero size if possible.")
                 # Ensure partial_size is at least 1 if p > 0 and dim > 0, to avoid splitting issues
                 partial_size = 1

            if partial_size > self.dim:
                 # This should ideally be caught by self.p > 1.0 check, but as a safeguard
                 partial_size = self.dim
                 logger.warning(f"Calculated partial_size ({partial_size}) exceeds dim ({self.dim}). "
                                f"Using full dim for normalization.")


        if partial_size == 0:
            logger.info("Partial size is 0. No normalization applied.")
            x_normed = x # Return the original tensor as no part is normalized
        else:
            if partial_size == self.dim:
                 norm_tensor = x
                 norm_dim_size = self.dim
            else:
                 norm_tensor, rest_tensor = torch.split(x, [partial_size, self.dim - partial_size], dim=-1)
                 norm_dim_size = partial_size

            rms_val = norm_tensor.norm(2, dim=-1, keepdim=True) * (norm_dim_size ** (-0.5)) # Use **-0.5 for clarity


            # Normalize the input tensor or the partial tensor
            normed_partial = norm_tensor / (rms_val + self.eps)

            # Recombine the normalized part and the rest of the tensor
            if partial_size == self.dim:
                 x_normed = normed_partial
            else:
                 x_normed = torch.cat([normed_partial, rest_tensor], dim=-1)


        if self.bias:
            return self.scale * x_normed + self.offset
        else:
            return self.scale * x_normed


if __name__ == "__main__":

    # Test cases
    input_tensor_dim = 512
    batch_size = 4
    seq_len = 10

    # 1. Full RMSNorm (default p=-1.0)
    print("\n--- Test Case 1: Full RMSNorm (default) ---")
    rms_norm_full = RMSNorm(dim=input_tensor_dim)
    x_full = torch.randn(batch_size, seq_len, input_tensor_dim)
    output_full = rms_norm_full(x_full)
    print(f"Input shape: {x_full.shape}, Output shape: {output_full.shape}")
    normed_part_full = output_full / rms_norm_full.scale
    sample_rms = normed_part_full[0, 0].norm(2) / (input_tensor_dim**0.5)
    print(f"RMS of a sample normalized output (should be close to 1): {sample_rms.item():.4f}")



    # 2. Partial RMSNorm (p=0.5)
    print("\n--- Test Case 2: Partial RMSNorm (p=0.5) ---")
    rms_norm_partial = RMSNorm(dim=input_tensor_dim, p=0.5)
    x_partial = torch.randn(batch_size, seq_len, input_tensor_dim)
    output_partial = rms_norm_partial(x_partial)
    print(f"Input shape: {x_partial.shape}, Output shape: {output_partial.shape}")
    partial_size_test2 = int(input_tensor_dim * 0.5)
  
  
    # 3. Full RMSNorm (p=1.0 - equivalent to full)
    print("\n--- Test Case 3: Full RMSNorm (p=1.0) ---")
    rms_norm_p1 = RMSNorm(dim=input_tensor_dim, p=1.0)
    x_p1 = torch.randn(batch_size, seq_len, input_tensor_dim)
    output_p1 = rms_norm_p1(x_p1)
    print(f"Input shape: {x_p1.shape}, Output shape: {output_p1.shape}")
    normed_part_p1 = output_p1 / rms_norm_p1.scale
    sample_rms_p1 = normed_part_p1[0, 0].norm(2) / (input_tensor_dim**0.5)
    print(f"RMS of a sample normalized output (should be close to 1): {sample_rms_p1.item():.4f}")


    # 4. RMSNorm with bias
    print("\n--- Test Case 4: RMSNorm with Bias ---")
    rms_norm_bias = RMSNorm(dim=input_tensor_dim, bias=True)
    x_bias = torch.randn(batch_size, seq_len, input_tensor_dim)
    output_bias = rms_norm_bias(x_bias)
    print(f"Input shape: {x_bias.shape}, Output shape: {output_bias.shape}")
    print(f"Scale parameter: {rms_norm_bias.scale.data[:5]}")
    print(f"Offset parameter: {rms_norm_bias.offset.data[:5]}")
    normed_part_bias = (output_bias - rms_norm_bias.offset) / rms_norm_bias.scale
    sample_rms_bias = normed_part_bias[0, 0].norm(2) / (input_tensor_dim**0.5)
    print(f"RMS of a sample normalized output (should be close to 1): {sample_rms_bias.item():.4f}")



    # 5. Edge case: p=0.0
    print("\n--- Test Case 5: Partial RMSNorm (p=0.0) - should apply no normalization to any dimension ---")
    rms_norm_p0 = RMSNorm(dim=input_tensor_dim, p=0.0)
    x_p0 = torch.randn(batch_size, seq_len, input_tensor_dim)
    output_p0 = rms_norm_p0(x_p0)
    print(f"Input shape: {x_p0.shape}, Output shape: {output_p0.shape}")
    print(f"Are output_p0 and x_p0 scaled by rms_norm_p0.scale close? {torch.allclose(output_p0, x_p0 * rms_norm_p0.scale)}")



    # 6. Error handling: invalid dim
    print("\n--- Test Case 6: Error - Invalid dim ---")
    try:
        RMSNorm(dim=0)
    except ValueError as e:
        print(f"Caught expected error: {e}")

    # 7. Error handling: input dim mismatch
    print("\n--- Test Case 7: Error - Input dim mismatch ---")
    rms_norm_mismatch = RMSNorm(dim=128)
    x_mismatch = torch.randn(batch_size, seq_len, 256)
    try:
        rms_norm_mismatch(x_mismatch)
    except ValueError as e:
        print(f"Caught expected error: {e}")

    # 8. Edge case: smaller dim and partial norm
    print("\n--- Test Case 8: Smaller dim and partial norm ---")
    rms_norm_small = RMSNorm(dim=10, p=0.3) # partial_size will be 3
    x_small = torch.randn(2, 5, 10)
    output_small = rms_norm_small(x_small)
    print(f"Input shape: {x_small.shape}, Output shape: {output_small.shape}")
    partial_size_test8 = int(10 * 0.3)
    normed_part_small = output_small[:, :, :partial_size_test8] / rms_norm_small.scale[:partial_size_test8]
    rms_small_part = normed_part_small.norm(2, dim=-1, keepdim=True) * (partial_size_test8**(-0.5))
    print(f"RMS of the normalized part (should be close to 1): {rms_small_part[0, 0].item():.4f}")