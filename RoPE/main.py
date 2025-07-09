import torch
import torch.nn.functional as F
import math

# --- 1. Precompute Rotary Frequencies (theta_i and m*theta_i) ---
def precompute_simple_rope_frequencies(head_dim: int, max_seq_len: int, device: str, theta: float = 10000.0):
    """
    Precomputes the frequency terms for Rotary Positional Embeddings (RoPE).
    This function calculates `m * theta_i` for all positions `m` and all `theta_i` pairs.

    Args:
        head_dim (int): The dimension of a single attention head. Must be even.
        max_seq_len (int): The maximum sequence length for which to precompute frequencies.
        device (str): The device ('cpu' or 'cuda') to place the tensors on.
        theta (float): The base value for the frequency calculation (default 10000.0).

    Returns:
        torch.Tensor: A complex tensor of shape (max_seq_len, head_dim / 2)
                      containing `exp(i * m * theta_i)` values.
    """
    assert head_dim % 2 == 0, "Head dimension must be divisible by 2 for RoPE."

    # Calculate theta_i values: 10000^(-2i/head_dim)
    # This creates a tensor like [10000^0, 10000^(-2/dim), 10000^(-4/dim), ...]
    # Shape: (head_dim / 2)
    theta_numerator = torch.arange(0, head_dim, 2).float()
    theta_inv = 1.0 / (theta ** (theta_numerator / head_dim)).to(device)

    # Create position indices: [0, 1, ..., max_seq_len-1]
    # Shape: (max_seq_len)
    m = torch.arange(max_seq_len, device=device)

    # Compute the outer product of positions and theta_inv
    # This gives us m * theta_i for each (position, frequency_component) pair.
    # Shape: (max_seq_len, head_dim / 2)
    freqs = torch.outer(m, theta_inv).float()

    # Convert frequencies to complex numbers in polar form: cos(freqs) + i * sin(freqs)
    # This is equivalent to e^(i * freqs)
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex

# --- 2. Apply Rotary Embeddings to a Tensor ---
def apply_simple_rope_torch(x: torch.Tensor, freqs_complex: torch.Tensor, position_indices: torch.Tensor, device: str):
    """
    Applies Rotary Positional Embeddings (RoPE) to the input tensor.

    Args:
        x (torch.Tensor): Input tensor (e.g., query or key vectors).
                          Expected shape: (batch_size, seq_len, num_heads, head_dim).
        freqs_complex (torch.Tensor): Precomputed complex frequencies from `precompute_simple_rope_frequencies`.
                                     Shape: (max_seq_len, head_dim / 2).
        position_indices (torch.Tensor): A tensor indicating the absolute position of each token
                                         in the sequence. Shape: (seq_len,).
                                         For a single token at `start_pos`, this would be `torch.tensor([start_pos])`.
        device (str): The device ('cpu' or 'cuda') to perform computations on.

    Returns:
        torch.Tensor: The input tensor `x` with RoPE applied. Same shape as `x`.
    """
    # Ensure x is on the correct device
    x = x.to(device)

    # Reshape x to treat pairs of elements along the head_dim as real and imaginary parts.
    # From (B, S, H, D) to (B, S, H, D/2, 2) then view as complex (B, S, H, D/2)
    # Example: (1, 1, 1, 4) -> (1, 1, 1, 2, 2) -> (1, 1, 1, 2) complex
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))

    # Select the relevant frequencies for the current positions
    # freqs_complex_selected shape: (seq_len, head_dim / 2)
    freqs_complex_selected = freqs_complex[position_indices].unsqueeze(0).unsqueeze(2)
    # After unsqueeze: (1, seq_len, 1, head_dim / 2) for broadcasting with x_complex

    # Apply the rotation by complex multiplication
    # (B, S, H, D/2) * (1, S, 1, D/2) -> (B, S, H, D/2)
    x_rotated_complex = x_complex * freqs_complex_selected

    # Convert the rotated complex tensor back to real numbers
    # From (B, S, H, D/2) complex to (B, S, H, D/2, 2) real
    x_out = torch.view_as_real(x_rotated_complex)

    # Reshape back to the original input shape (B, S, H, D)
    x_out = x_out.reshape(*x.shape)

    # Ensure the output tensor has the same data type as the original input
    return x_out.type_as(x)


# --- Test Case ---
if __name__ == "__main__":
    # Define parameters for a simple test
    batch_size = 1
    seq_len = 1  # For token-by-token processing
    num_heads = 1
    head_dim = 4  # Must be even for RoPE (e.g., 4, 8, 16, 64)
    max_seq_len = 10 # Max sequence length for precomputing frequencies
    device = "cpu" # "cuda" if you have a GPU

    print(f"--- Simple RoPE Test with PyTorch ---")
    print(f"Device: {device}")
    print(f"Head Dimension: {head_dim}")
    print(f"Max Sequence Length for Frequencies: {max_seq_len}")

    # Precompute frequencies once
    precomputed_freqs = precompute_simple_rope_frequencies(head_dim, max_seq_len, device)
    print(f"\nPrecomputed frequencies shape: {precomputed_freqs.shape}") # Expected: (max_seq_len, head_dim/2)

    # Create a dummy original embedding vector (e.g., a single token's Q or K)
    # Shape: (batch_size, seq_len, num_heads, head_dim)
    # Example: (1, 1, 1, 4)
    original_embedding = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]], dtype=torch.float32).to(device)
    print(f"\nOriginal embedding (shape {original_embedding.shape}):\n{original_embedding.squeeze()}")

    # Apply rotary embedding for different positions
    for position in range(5): # Test positions 0 to 4
        # Get the position index tensor (for a single token)
        position_indices = torch.tensor([position], dtype=torch.long).to(device)

        # Apply RoPE
        rotated_embedding = apply_simple_rope_torch(
            original_embedding,
            precomputed_freqs,
            position_indices,
            device
        )

        print(f"\n--- Position {position} ---")
        print(f"Rotated embedding (shape {rotated_embedding.shape}):\n{rotated_embedding.squeeze()}")

        # Verify that the L2 norm (magnitude) of the vector remains unchanged
        # RoPE is an orthogonal transformation, so it preserves vector magnitude.
        original_norm = torch.norm(original_embedding)
        rotated_norm = torch.norm(rotated_embedding)
        print(f"Original norm: {original_norm:.6f}, Rotated norm: {rotated_norm:.6f}")
        assert torch.isclose(original_norm, rotated_norm, atol=1e-5), \
            f"Norm mismatch! Original: {original_norm}, Rotated: {rotated_norm}"
        print("Norm preserved (as expected for RoPE).")

    # --- Test with different head_dim and batch_size ---
    print("\n--- Testing with different head_dim and batch_size (GQA-like scenario) ---")
    test_head_dim = 64
    test_num_heads = 4
    test_batch_size = 2
    test_max_seq_len = 5

    test_args = {
        'head_dim': test_head_dim,
        'max_seq_len': test_max_seq_len,
        'device': device
    }

    test_precomputed_freqs = precompute_simple_rope_frequencies(**test_args)

    # Create a dummy batch of embeddings
    test_embeddings = torch.randn(
        test_batch_size, seq_len, test_num_heads, test_head_dim, dtype=torch.float32
    ).to(device)
    print(f"\nTest embeddings shape: {test_embeddings.shape}")

    for position in range(test_max_seq_len):
        test_position_indices = torch.tensor([position], dtype=torch.long).to(device)
        rotated_test_embeddings = apply_simple_rope_torch(
            test_embeddings,
            test_precomputed_freqs,
            test_position_indices,
            device
        )
        print(f"Position {position}: Rotated test embeddings shape: {rotated_test_embeddings.shape}")
        assert rotated_test_embeddings.shape == test_embeddings.shape
        # Check norm preservation for the batch
        assert torch.allclose(torch.norm(test_embeddings, dim=-1), torch.norm(rotated_test_embeddings, dim=-1), atol=1e-5)
        print(f"Norms preserved for batch at position {position}.")

    print("\nAll simple RoPE tests completed successfully!")

