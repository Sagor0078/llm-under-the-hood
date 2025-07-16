import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class SlidingWindowAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, window_size, is_causal=True):
        super().__init__()
        """
        Conceptual implementation of a single Sliding Window Attention layer.

        Args:
            embed_dim (int): The embedding dimension of the input.
            num_heads (int): The number of attention heads.
            window_size (int): The size of the sliding window (e.g., 3 for current + 1 left + 1 right).
                               For causal attention, this is the number of tokens to look back.
            is_causal (bool): If True, applies a causal mask (only attends to past tokens).
        """
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.window_size = window_size
        self.is_causal = is_causal

        # Linear projections for Query, Key, Value
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def _split_heads(self, x, batch_size):
        """Splits the input into multiple heads."""
        # x: (batch_size, seq_len, embed_dim)
        return x.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        # Output: (batch_size, num_heads, seq_len, head_dim)

    def _combine_heads(self, x, batch_size):
        """Combines multiple heads back to original embedding dimension."""
        # x: (batch_size, num_heads, seq_len, head_dim)
        return x.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        # Output: (batch_size, seq_len, embed_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        # Project Q, K, V
        q = self.q_proj(x) # (batch_size, seq_len, embed_dim)
        k = self.k_proj(x) # (batch_size, seq_len, embed_dim)
        v = self.v_proj(x) # (batch_size, seq_len, embed_dim)

        # Split into heads
        q = self._split_heads(q, batch_size) # (batch_size, num_heads, seq_len, head_dim)
        k = self._split_heads(k, batch_size) # (batch_size, num_heads, seq_len, head_dim)
        v = self._split_heads(v, batch_size) # (batch_size, num_heads, seq_len, head_dim)

        # Calculate attention scores (Q @ K_T)
        # (batch_size, num_heads, seq_len, seq_len)
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Apply Sliding Window Mask 
        # Create a mask that allows attention only within the window
        # For each query at position `i`, it can attend to keys from `i - window_size + 1` to `i`.
        # If not causal, it attends from `i - window_size/2` to `i + window_size/2`.

        mask = torch.full((seq_len, seq_len), float('-inf'), device=x.device)

        if self.is_causal:
            # Causal mask + sliding window constraint
            for i in range(seq_len):
                # Attend to tokens from max(0, i - window_size + 1) to i
                start_idx = max(0, i - self.window_size + 1)
                mask[i, start_idx : i + 1] = 0.0 # Allow attention
        else:
            # Non-causal sliding window mask
            half_window = self.window_size // 2
            for i in range(seq_len):
                start_idx = max(0, i - half_window)
                end_idx = min(seq_len, i + half_window + 1)
                mask[i, start_idx:end_idx] = 0.0

        # Apply the mask to attention scores
        # The mask needs to be broadcastable to (batch_size, num_heads, seq_len, seq_len)
        attention_scores = attention_scores + mask.unsqueeze(0).unsqueeze(0)

        # Apply softmax to get attention probabilities
        attention_probs = F.softmax(attention_scores, dim=-1)

        # Apply attention probabilities to values
        # (batch_size, num_heads, seq_len, head_dim)
        context_layer = torch.matmul(attention_probs, v)

        # Combine heads
        # (batch_size, seq_len, embed_dim)
        context_layer = self._combine_heads(context_layer, batch_size)

        # Final linear projection
        output = self.out_proj(context_layer)
        return output

# Example Usage 
if __name__ == "__main__":
    # Define model parameters
    EMBED_DIM = 256
    NUM_HEADS = 8
    WINDOW_SIZE = 5 # Each token attends to itself and 4 preceding tokens (for causal)
    SEQ_LEN = 20
    BATCH_SIZE = 2

    print(f"Sliding Window Attention (Causal) Demo")
    print(f"Embed Dim: {EMBED_DIM}, Num Heads: {NUM_HEADS}, Window Size: {WINDOW_SIZE}")
    print(f"Sequence Length: {SEQ_LEN}, Batch Size: {BATCH_SIZE}")

    # Create a dummy input tensor
    # (batch_size, seq_len, embed_dim)
    dummy_input = torch.randn(BATCH_SIZE, SEQ_LEN, EMBED_DIM)

    # Instantiate the SlidingWindowAttention module (causal)
    causal_swa_layer = SlidingWindowAttention(
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        window_size=WINDOW_SIZE,
        is_causal=True
    )

    print("\nRunning causal SWA forward pass...")
    output_causal = causal_swa_layer(dummy_input)
    print(f"Output shape (causal): {output_causal.shape}")
    print("This output represents the contextualized embeddings after one SWA layer.")

    # Demonstrate the mask 
    print("\nVisualizing the causal sliding window mask for SEQ_LEN=10, WINDOW_SIZE=3:")
    temp_seq_len = 10
    temp_window_size = 3
    temp_mask = torch.full((temp_seq_len, temp_seq_len), float('-inf'))
    for i in range(temp_seq_len):
        start_idx = max(0, i - temp_window_size + 1)
        temp_mask[i, start_idx : i + 1] = 0.0
    
    # Replace -inf with a more readable character for visualization
    mask_display = np.where(temp_mask.cpu().numpy() == 0.0, '0', 'X')
    print("  (0 = allowed attention, X = masked)")
    for row in mask_display:
        print(" ".join(row))

    print("\n Sliding Window Attention (Non-Causal) Demo ")
    # Instantiate the SlidingWindowAttention module (non-causal)
    non_causal_swa_layer = SlidingWindowAttention(
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        window_size=WINDOW_SIZE, # Here, window_size includes current + half_left + half_right
        is_causal=False
    )

    print("\nRunning non-causal SWA forward pass...")
    output_non_causal = non_causal_swa_layer(dummy_input)
    print(f"Output shape (non-causal): {output_non_causal.shape}")
    
    # Demonstrate the non-causal mask 
    print("\nVisualizing the non-causal sliding window mask for SEQ_LEN=10, WINDOW_SIZE=5:")
    temp_seq_len = 10
    temp_window_size = 5
    temp_mask_non_causal = torch.full((temp_seq_len, temp_seq_len), float('-inf'))
    half_window = temp_window_size // 2
    for i in range(temp_seq_len):
        start_idx = max(0, i - half_window)
        end_idx = min(temp_seq_len, i + half_window + 1)
        temp_mask_non_causal[i, start_idx:end_idx] = 0.0
    
    mask_display_non_causal = np.where(temp_mask_non_causal.cpu().numpy() == 0.0, '0', 'X')
    print("  (0 = allowed attention, X = masked)")
    for row in mask_display_non_causal:
        print(" ".join(row))

    print("\nNote: This is a single layer. In a full LLM, multiple such layers would be stacked,")
    print("expanding the effective receptive field.")
