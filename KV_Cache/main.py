''' source code refactored by LLM '''

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32  # Number of heads for the queries
    n_kv_heads: Optional[int] = None  # Number of heads for the keys and values
    vocab_size: int = -1  # This will be set during tokenizer
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    max_batch_size: int = 1  # Added for KV cache pre-allocation
    max_seq_len: int = 2048  # Added for KV cache pre-allocation
    device: str = "cpu"  # Added to specify device for tensors


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        # Calculate the root mean square of the input tensor along the last dimension
        # Add epsilon for numerical stability
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        # Apply RMS normalization and scale by the learnable weight
        # Convert to float for computation and then back to original type
        return self.weight * self._norm(x.float()).type_as(x)


def precompute_theta_pos_frequencies(head_dim: int, seq_len: int, device: str, theta: float = 10000.0):
    """
    Precomputes the theta positional frequencies for Rotary Positional Embeddings (RoPE).
    These frequencies are used to apply rotation to query and key vectors.
    """
    assert head_dim % 2 == 0, "Dimension must be divisible by 2 for RoPE"

    # Create the theta parameters based on the formula: theta_i = 10000^(-2i/head_dim)
    # Shape: (head_dim / 2)
    theta_numerator = torch.arange(0, head_dim, 2).float()
    theta_inv = 1.0 / (theta ** (theta_numerator / head_dim)).to(device)

    # Create position indices: m = [0, 1, ..., seq_len-1]
    m = torch.arange(seq_len, device=device)

    # Compute the outer product of positions and theta_inv to get frequencies
    # freqs[p, i] = m[p] * theta_inv[i]
    # Shape: (seq_len, head_dim / 2)
    freqs = torch.outer(m, theta_inv).float()

    # Convert frequencies to complex numbers in polar form: e^(i * freqs)
    # This is equivalent to cos(freqs) + i * sin(freqs)
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex


def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor, device: str):
    """
    Applies Rotary Positional Embeddings (RoPE) to the input tensor.
    Args:
        x (torch.Tensor): Input tensor (query or key) of shape (B, seq_len, H, Head_dim).
        freqs_complex (torch.Tensor): Precomputed complex frequencies of shape (seq_len, Head_dim/2).
        device (str): The device to perform computations on.
    Returns:
        torch.Tensor: The input tensor with RoPE applied, same shape as x.
    """
    # Reshape x to (B, seq_len, H, Head_dim/2, 2) to treat pairs as real/imaginary parts
    # Then view as complex numbers: (B, seq_len, H, Head_dim/2)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))

    # Reshape freqs_complex for broadcasting: (1, seq_len, 1, Head_dim/2)
    # This aligns with (B, seq_len, H, Head_dim/2) for element-wise multiplication
    freqs_complex_reshaped = freqs_complex.unsqueeze(0).unsqueeze(2)

    # Apply the rotation by multiplying the complex tensors
    x_rotated = x_complex * freqs_complex_reshaped

    # Convert the rotated complex tensor back to real numbers
    # Shape: (B, seq_len, H, Head_dim/2, 2)
    x_out = torch.view_as_real(x_rotated)

    # Reshape back to the original input shape: (B, seq_len, H, Head_dim)
    x_out = x_out.reshape(*x.shape)

    # Ensure the output tensor has the same data type as the original input and is on the correct device
    return x_out.type_as(x).to(device)


def repeat_kv(x: torch.Tensor, n_repeats: int) -> torch.Tensor:
    """
    Repeats the Key and Value heads for Grouped-Query Attention (GQA).
    If n_repeats is 1 (MHA), the tensor is returned as is.
    If n_repeats > 1 (GQA), K/V heads are duplicated to match the number of query heads.
    Args:
        x (torch.Tensor): Tensor of shape (batch_size, seq_len, n_kv_heads, head_dim)
        n_repeats (int): Number of times to repeat each KV head.
    Returns:
        torch.Tensor: Tensor with repeated KV heads, shape (batch_size, seq_len, n_q_heads, head_dim)
    """
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    if n_repeats == 1:
        return x
    return (
        # Add a new dimension to expand along: (B, seq_len, n_kv_heads, 1, head_dim)
        x[:, :, :, None, :]
        # Expand along the new dimension by n_repeats: (B, seq_len, n_kv_heads, n_repeats, head_dim)
        .expand(batch_size, seq_len, n_kv_heads, n_repeats, head_dim)
        # Reshape to combine n_kv_heads and n_repeats into the new n_q_heads dimension
        # (B, seq_len, n_kv_heads * n_repeats, head_dim)
        .reshape(batch_size, seq_len, n_kv_heads * n_repeats, head_dim)
    )


class FeedForward(nn.Module):
    def __init__(
            self,
            args: ModelArgs
    ):
        super().__init__()

        # Calculate hidden dimension for the feed-forward network
        # Starts with 4 * dim, then scaled by 2/3, then by ffn_dim_multiplier if provided
        hidden_dim = 4 * args.dim
        hidden_dim = int(2 * hidden_dim / 3)
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)

        # Round up hidden_dim to the nearest multiple_of for efficiency
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)

        # Linear layers for the SwiGLU-like activation
        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)  # Gate projection
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)  # Output projection
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)  # Value projection

    def forward(self, x: torch.Tensor):
        # SwiGLU activation: F.silu(w1(x)) * w3(x)
        swish_gate = F.silu(self.w1(x))
        value_proj = self.w3(x)
        gated_output = swish_gate * value_proj
        # Final linear projection
        x = self.w2(gated_output)
        return x


class MultiHeadAttentionBlock(nn.Module):
    def __init__(
            self,
            args: ModelArgs
    ):
        super().__init__()

        # Determine the number of key/value heads. Defaults to query heads if not specified (MHA).
        # Otherwise, it's Grouped-Query Attention (GQA).
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_q_heads = args.n_heads  # Number of query heads
        # Calculate how many times KV heads need to be repeated for GQA
        self.n_repeats = self.n_q_heads // self.n_kv_heads
        # Calculate the dimension of each attention head
        self.head_dim = args.dim // args.n_heads

        # Linear layers for Query, Key, Value, and Output projections
        # wq projects to n_q_heads * head_dim
        self.wq = nn.Linear(args.dim, self.n_q_heads * self.head_dim, bias=False)
        # wk and wv project to n_kv_heads * head_dim (for GQA/MQA efficiency)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        # wo projects the concatenated head outputs back to the model's dimension
        self.wo = nn.Linear(self.n_q_heads * self.head_dim, args.dim, bias=False)

        # KV Cache: Pre-allocate tensors to store keys and values from previous tokens
        # This avoids recomputing them for each new token during autoregressive generation.
        # Shape: (max_batch_size, max_seq_len, n_kv_heads, head_dim)
        self.cache_keys = torch.zeros(
            (args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim)
        ).to(args.device) # Ensure cache is on the correct device
        self.cache_values = torch.zeros(
            (args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim)
        ).to(args.device) # Ensure cache is on the correct device

    @staticmethod
    def attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, head_dim: int):
        """
        Performs scaled dot-product attention.
        Args:
            query (torch.Tensor): Query tensor, shape (B, H_Q, Seq_Len_Q, Head_Dim)
            key (torch.Tensor): Key tensor, shape (B, H_KV, Seq_Len_KV, Head_Dim)
            value (torch.Tensor): Value tensor, shape (B, H_KV, Seq_Len_KV, Head_Dim)
            head_dim (int): Dimension of each attention head.
        Returns:
            torch.Tensor: Output of the attention mechanism, shape (B, H_Q, Seq_Len_Q, Head_Dim)
        """
        # Calculate attention scores: (Query @ Key.T) / sqrt(head_dim)
        # Shape: (B, H_Q, Seq_Len_Q, Seq_Len_KV)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(head_dim)
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1).type_as(query)
        # Multiply weights by values to get the context output
        output = attention_weights @ value
        return output

    def forward(
            self,
            x: torch.Tensor,
            start_pos: int,
            freqs_complex: torch.Tensor
    ):
        """
        Forward pass for the MultiHeadAttentionBlock.
        Args:
            x (torch.Tensor): Input tensor for the current token, shape (batch_size, 1, dim).
            start_pos (int): The starting position of the current token in the sequence.
            freqs_complex (torch.Tensor): Precomputed RoPE frequencies for the current token,
                                         shape (1, head_dim/2).
        Returns:
            torch.Tensor: Output of the attention block, shape (batch_size, 1, dim).
        """
        batch_size, seq_len, _ = x.shape  # (batch_size, 1, dim) due to token-by-token generation

        # Project input x to Query, Key, and Value
        # (B, 1, Dim) -> (B, 1, H_Q * Head_Dim)
        query = self.wq(x)
        # (B, 1, Dim) -> (B, 1, H_KV * Head_Dim)
        key = self.wk(x)
        # (B, 1, Dim) -> (B, 1, H_KV * Head_Dim)
        value = self.wv(x)

        # Reshape Q, K, V to separate heads
        # (B, 1, H_Q * Head_Dim) -> (B, 1, H_Q, Head_Dim)
        query = query.view(batch_size, seq_len, self.n_q_heads, self.head_dim)
        # (B, 1, H_KV * Head_Dim) -> (B, 1, H_KV, Head_Dim)
        key = key.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        # (B, 1, H_KV * Head_Dim) -> (B, 1, H_KV, Head_Dim)
        value = value.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        # Apply Rotary Positional Embeddings to Query and Key
        # freqs_complex here should be the slice corresponding to the current token's position
        query = apply_rotary_embeddings(query, freqs_complex, device=x.device)
        key = apply_rotary_embeddings(key, freqs_complex, device=x.device)

        # Store the newly computed Key and Value into the KV cache
        # The cache is pre-allocated, so we just write to the current position
        # Slicing: cache_keys[batch_idx, current_token_pos : current_token_pos + 1]
        self.cache_keys[:batch_size, start_pos: start_pos + seq_len] = key
        self.cache_values[:batch_size, start_pos: start_pos + seq_len] = value

        # Retrieve all relevant Keys and Values from the cache so far
        # This includes keys/values from all previous tokens + the current token
        # Shape: (batch_size, start_pos + seq_len, n_kv_heads, head_dim)
        keys_from_cache = self.cache_keys[:batch_size, 0:start_pos + seq_len]
        values_from_cache = self.cache_values[:batch_size, 0:start_pos + seq_len]

        # Apply repeat_kv for Grouped-Query Attention (GQA) if n_repeats > 1
        # This aligns the number of KV heads with the number of Query heads for attention calculation
        keys_expanded = repeat_kv(keys_from_cache, self.n_repeats)
        values_expanded = repeat_kv(values_from_cache, self.n_repeats)

        # Transpose query, expanded keys, and expanded values for batch matrix multiplication
        # Shape: (B, Num_Heads, Seq_Len, Head_Dim)
        query = query.transpose(1, 2)
        keys_expanded = keys_expanded.transpose(1, 2)
        values_expanded = values_expanded.transpose(1, 2)

        # Perform attention calculation
        output = MultiHeadAttentionBlock.attention(query, keys_expanded, values_expanded, self.head_dim)

        # Reshape the output back to (batch_size, seq_len, dim)
        # .contiguous() is important after transpose to ensure memory contiguity before view
        output = (output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1))

        # Final linear projection
        return self.wo(output)


class EncoderBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads

        self.attention = MultiHeadAttentionBlock(args)
        self.feed_forward = FeedForward(args)

        # Normalization before attention block (pre-normalization)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        # Normalization before feed-forward block (pre-normalization)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        """
        Forward pass for a single Transformer Encoder Block.
        Applies attention, residual connection, normalization, feed-forward, and another residual connection.
        """
        # Attention sub-layer with pre-normalization and residual connection
        h = x + self.attention.forward(
            self.attention_norm(x), start_pos, freqs_complex
        )

        # Feed-forward sub-layer with pre-normalization and residual connection
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        assert args.vocab_size != -1, "Vocab size must be set in ModelArgs"

        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers

        # Token embeddings layer
        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim)

        # Stack of Encoder (Transformer) blocks
        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(EncoderBlock(args))

        # Final RMS normalization layer
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        # Output linear layer to project back to vocabulary size for token prediction
        self.output = nn.Linear(args.dim, self.vocab_size, bias=False)

        # Precompute RoPE frequencies once for the maximum sequence length
        self.freqs_complex = precompute_theta_pos_frequencies(
            self.args.dim // self.args.n_heads,
            self.args.max_seq_len,
            device=self.args.device
        )

    def forward(self, tokens: torch.Tensor, start_position: int):
        """
        Forward pass for the Transformer model.
        Processes tokens one by one in an autoregressive manner.
        Args:
            tokens (torch.Tensor): Input token IDs, shape (batch_size, 1).
            start_position (int): The current position of the token in the sequence (0 for first, 1 for second, etc.).
        Returns:
            torch.Tensor: Logits for the next token prediction, shape (batch_size, 1, vocab_size).
        """
        # Input tokens are expected to be (batch_size, 1) for token-by-token generation
        batch_size, seq_len = tokens.size()
        assert seq_len == 1, "This Transformer implementation is designed for token-by-token generation (seq_len must be 1)"

        # Convert token IDs to embeddings
        # Shape: (B, 1) -> (B, 1, Dim)
        h = self.tok_embeddings(tokens)

        # Retrieve the relevant RoPE frequencies for the current position
        # Shape: (1, head_dim/2)
        current_freqs_complex = self.freqs_complex[start_position : start_position + seq_len]

        # Pass through each EncoderBlock (Transformer layer)
        for layer in self.layers:
            h = layer(h, start_position, current_freqs_complex)

        # Apply final normalization
        h = self.norm(h)

        # Project to vocabulary size to get logits
        output = self.output(h).float()
        return output

# --- Test Case ---
if __name__ == "__main__":
    def run_inference_test(test_name: str, args: ModelArgs):
        print(f"\n--- Running Test Scenario: {test_name} ---")
        print(f"Model Args: dim={args.dim}, n_heads={args.n_heads}, n_kv_heads={args.n_kv_heads}, "
              f"max_batch_size={args.max_batch_size}, max_seq_len={args.max_seq_len}, device={args.device}")

        model = Transformer(args)
        
        # Access the attention block of the first layer for KV cache inspection
        first_attention_block = model.layers[0].attention

        for i in range(args.max_seq_len):
            # Create dummy input tokens for the batch
            dummy_tokens = torch.tensor(
                [[ (i + j) % args.vocab_size for _ in range(1)] for j in range(args.max_batch_size)],
                dtype=torch.long
            ).to(args.device)
            current_position = i

            print(f"\n--- Generating token at position {current_position} ---")
            print(f"Input tokens (batch): {dummy_tokens.squeeze().tolist()}")

            try:
                output_logits = model.forward(dummy_tokens, current_position)
                print(f"Output logits shape: {output_logits.shape}")
                assert output_logits.shape == (args.max_batch_size, 1, args.vocab_size)
                print("Forward pass successful.")

                # --- KV Cache Inspection ---
                # Check the shape of the retrieved keys and values from cache
                # They should grow with each step
                expected_cached_seq_len = current_position + 1
                
                # Retrieve the actual cached keys/values for the current batch
                cached_keys_current_step = first_attention_block.cache_keys[:args.max_batch_size, :expected_cached_seq_len]
                cached_values_current_step = first_attention_block.cache_values[:args.max_batch_size, :expected_cached_seq_len]

                print(f"Cached keys shape (Layer 0): {cached_keys_current_step.shape}")
                print(f"Cached values shape (Layer 0): {cached_values_current_step.shape}")
                
                # Assert that the cached sequence length is as expected
                assert cached_keys_current_step.shape[1] == expected_cached_seq_len
                assert cached_values_current_step.shape[1] == expected_cached_seq_len

                # Assert that the current position in the cache is not all zeros (indicates it was written to)
                # This is a basic check, a more rigorous test would compare exact values if possible.
                if current_position == 0: # For the first token, the cache is just this token
                    assert not torch.all(first_attention_block.cache_keys[:args.max_batch_size, current_position].eq(0))
                    assert not torch.all(first_attention_block.cache_values[:args.max_batch_size, current_position].eq(0))
                else: # For subsequent tokens, check the newly written part
                    assert not torch.all(first_attention_block.cache_keys[:args.max_batch_size, current_position].eq(0))
                    assert not torch.all(first_attention_block.cache_values[:args.max_batch_size, current_position].eq(0))
                
                print(f"KV cache for position {current_position} appears to be populated.")

                # Simulate picking the next token
                predicted_token_ids = torch.argmax(output_logits, dim=-1)
                print(f"Predicted next token IDs (for this step): {predicted_token_ids.squeeze().tolist()}")

            except Exception as e:
                print(f"An error occurred during forward pass at position {current_position}: {e}")
                break
        print(f"--- Test Scenario '{test_name}' finished. ---")


    # --- Test Scenarios ---

    # Scenario 1: Basic GQA (n_heads=8, n_kv_heads=4, batch_size=2)
    args_gqa_batch2 = ModelArgs(
        dim=512,
        n_layers=2,
        n_heads=8,
        n_kv_heads=4,
        vocab_size=1000,
        max_batch_size=2,
        max_seq_len=5, # Shorter sequence for quicker test
        device="cpu"
    )
    run_inference_test("GQA (n_kv_heads < n_heads) with Batch Size 2", args_gqa_batch2)

    # Scenario 2: MHA (n_kv_heads = n_heads, batch_size=1)
    args_mha_batch1 = ModelArgs(
        dim=512,
        n_layers=2,
        n_heads=8,
        n_kv_heads=8, # MHA configuration
        vocab_size=1000,
        max_batch_size=1,
        max_seq_len=5,
        device="cpu"
    )
    run_inference_test("MHA (n_kv_heads = n_heads) with Batch Size 1", args_mha_batch1)

    # Scenario 3: GQA with larger batch size (n_heads=8, n_kv_heads=2, batch_size=4)
    args_gqa_batch4 = ModelArgs(
        dim=512,
        n_layers=2,
        n_heads=8,
        n_kv_heads=2, # More aggressive GQA
        vocab_size=1000,
        max_batch_size=4,
        max_seq_len=5,
        device="cpu"
    )
    run_inference_test("GQA (n_kv_heads < n_heads) with Batch Size 4", args_gqa_batch4)

    print("\nAll test scenarios completed.")

