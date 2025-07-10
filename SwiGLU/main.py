import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLUFeedForward(nn.Module):
    """
    Mathematically, SwiGLU(x) = (SiLU(xW1)) * (xW3) * W2,
    where '*' denotes element-wise multiplication (Hadamard product).
    """
    def __init__(self,
                 dim: int,
                 hidden_dim: int,
                 bias: bool = False):
        super().__init__()

        # This output will be passed through the SiLU activation.
        self.w1 = nn.Linear(dim, hidden_dim, bias=bias)

        # This output will be element-wise multiplied with the activated 'gate' pathway.
        self.w3 = nn.Linear(dim, hidden_dim, bias=bias)

        self.w2 = nn.Linear(hidden_dim, dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
       
        gate_output = F.silu(self.w1(x))

        # Apply the second linear transformation (w3) for the 'value' component.
        # Shape: (..., dim) -> (..., hidden_dim)
        value_output = self.w3(x)

        # Perform element-wise multiplication between the 'gate' and 'value' components.
        # This is the core 'gating' operation of SwiGLU.
        # Shape: (..., hidden_dim) * (..., hidden_dim) -> (..., hidden_dim)
        gated_combined_output = gate_output * value_output

        # Apply the final linear transformation (w2) to project back to the original dimension.
        # Shape: (..., hidden_dim) -> (..., dim)
        final_output = self.w2(gated_combined_output)

        return final_output

# --- Test Case ---
if __name__ == "__main__":
    print("--- Testing SwiGLUFeedForward ---")

    input_dim = 512
    hidden_dim_multiplier = 4
    calculated_hidden_dim = input_dim * hidden_dim_multiplier

    # Create an instance of the SwiGLUFeedForward network
    swiglu_ffn = SwiGLUFeedForward(dim=input_dim, hidden_dim=calculated_hidden_dim, bias=False)
    print(f"SwiGLU FFN initialized with input_dim={input_dim}, hidden_dim={calculated_hidden_dim}")

    # Create a dummy input tensor
    # Batch size = 2, Sequence length = 10, Embedding dimension = input_dim
    batch_size = 2
    seq_len = 10
    dummy_input = torch.randn(batch_size, seq_len, input_dim)
    print(f"\nDummy input shape: {dummy_input.shape}")

    # Perform a forward pass
    output = swiglu_ffn(dummy_input)

    print(f"Output shape: {output.shape}")

    # Assert that the output shape is the same as the input shape
    assert output.shape == dummy_input.shape, "Output shape mismatch!"
    print("Output shape matches input shape (as expected).")

    # Verify that the output is not all zeros (basic sanity check)
    assert not torch.all(output == 0), "Output is all zeros, something might be wrong!"
    print("Output is not all zeros (basic sanity check passed).")

    print("\n--- SwiGLUFeedForward Test Completed Successfully! ---")

    # Example with a single token input (common in autoregressive decoding)
    print("\n--- Testing with single token input ---")
    single_token_input = torch.randn(1, 1, input_dim)
    print(f"Single token input shape: {single_token_input.shape}")
    single_token_output = swiglu_ffn(single_token_input)
    print(f"Single token output shape: {single_token_output.shape}")
    assert single_token_output.shape == single_token_input.shape
    print("Single token test passed.")

