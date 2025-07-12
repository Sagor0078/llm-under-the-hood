import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ScaledDotProductAttention(nn.Module):
    """
    Implements the core scaled dot-product attention mechanism.
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
    """
    def __init__(self, dropout_p: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            q (torch.Tensor): Query tensor (..., seq_len_q, head_dim)
            k (torch.Tensor): Key tensor (..., seq_len_kv, head_dim)
            v (torch.Tensor): Value tensor (..., seq_len_kv, head_dim)
            mask (torch.Tensor, optional): Attention mask (..., seq_len_q, seq_len_kv).
                                           Values of -inf (or very small negative) hide connections.
        Returns:
            torch.Tensor: Output tensor of attention (..., seq_len_q, head_dim)
            torch.Tensor: Attention weights (..., seq_len_q, seq_len_kv)
        """
        # Get head_dim from the last dimension of Q (or K, V)
        d_k = q.size(-1)

        # Compute attention scores: QK^T
        # (..., seq_len_q, head_dim) @ (..., head_dim, seq_len_kv) -> (..., seq_len_q, seq_len_kv)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

        # Apply optional mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf')) # Mask out positions

        # Apply softmax to get attention weights
        # Normalizes scores across the last dimension (seq_len_kv)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights) # Apply dropout

        # Multiply weights by Value matrix
        # (..., seq_len_q, seq_len_kv) @ (..., seq_len_kv, head_dim) -> (..., seq_len_q, head_dim)
        output = torch.matmul(attn_weights, v)

        return output, attn_weights

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention (MHA) as introduced in "Attention Is All You Need".
    Each head has its own independent Q, K, V linear projections.
    """
    def __init__(self, model_dim: int, num_heads: int, dropout_p: float = 0.0, bias: bool = False):
        super().__init__()
        if model_dim % num_heads != 0:
            raise ValueError(f"model_dim ({model_dim}) must be divisible by num_heads ({num_heads})")

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads

        # Linear projections for Queries, Keys, Values
        # Combined for efficiency: one large matrix multiplication then split
        self.q_proj = nn.Linear(model_dim, model_dim, bias=bias)
        self.k_proj = nn.Linear(model_dim, model_dim, bias=bias)
        self.v_proj = nn.Linear(model_dim, model_dim, bias=bias)

        # Output linear projection
        self.out_proj = nn.Linear(model_dim, model_dim, bias=bias)

        self.attention = ScaledDotProductAttention(dropout_p)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            x (torch.Tensor): Input tensor (batch_size, seq_len, model_dim)
            mask (torch.Tensor, optional): Attention mask (batch_size, 1, seq_len, seq_len).
                                           Typically a causal mask for decoder-only models.
        Returns:
            torch.Tensor: Output tensor (batch_size, seq_len, model_dim)
            torch.Tensor: Attention weights (batch_size, num_heads, seq_len, seq_len)
        """
        batch_size, seq_len, _ = x.size()

        # Apply linear projections and reshape for multi-head
        # (batch_size, seq_len, model_dim) -> (batch_size, seq_len, num_heads, head_dim)
        # Then transpose to (batch_size, num_heads, seq_len, head_dim) for attention
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Perform scaled dot-product attention
        # Output: (batch_size, num_heads, seq_len, head_dim)
        # Attn_weights: (batch_size, num_heads, seq_len, seq_len)
        attn_output, attn_weights = self.attention(q, k, v, mask)

        # Concatenate heads and apply final linear projection
        # (batch_size, num_heads, seq_len, head_dim) -> (batch_size, seq_len, num_heads, head_dim)
        # -> (batch_size, seq_len, model_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.model_dim)
        output = self.out_proj(attn_output)

        return output, attn_weights

class MultiQueryAttention(nn.Module):
    """
    Multi-Query Attention (MQA).
    All query heads share a single Key and a single Value projection.
    """
    def __init__(self, model_dim: int, num_heads: int, dropout_p: float = 0.0, bias: bool = False):
        super().__init__()
        if model_dim % num_heads != 0:
            raise ValueError(f"model_dim ({model_dim}) must be divisible by num_heads ({num_heads})")

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads

        # Q projection for all heads
        self.q_proj = nn.Linear(model_dim, model_dim, bias=bias)
        # Single K and V projection for all heads
        self.k_proj = nn.Linear(model_dim, self.head_dim, bias=bias) # Output dim is head_dim, not model_dim
        self.v_proj = nn.Linear(model_dim, self.head_dim, bias=bias) # Output dim is head_dim, not model_dim

        # Output linear projection (input is model_dim, as Q heads are concatenated)
        self.out_proj = nn.Linear(model_dim, model_dim, bias=bias)

        self.attention = ScaledDotProductAttention(dropout_p)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        batch_size, seq_len, _ = x.size()

        # Apply Q projection and reshape for multi-head
        # (batch_size, seq_len, model_dim) -> (batch_size, seq_len, num_heads, head_dim)
        # -> (batch_size, num_heads, seq_len, head_dim)
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply K and V projections (single head for K/V)
        # (batch_size, seq_len, model_dim) -> (batch_size, seq_len, head_dim)
        # -> (batch_size, 1, seq_len, head_dim) - unsqueeze for head dimension
        k = self.k_proj(x).view(batch_size, seq_len, 1, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, 1, self.head_dim).transpose(1, 2)

        # Repeat K and V across the head dimension for broadcasting
        # (batch_size, 1, seq_len, head_dim) -> (batch_size, num_heads, seq_len, head_dim)
        k = k.expand(-1, self.num_heads, -1, -1)
        v = v.expand(-1, self.num_heads, -1, -1)

        # Perform scaled dot-product attention
        attn_output, attn_weights = self.attention(q, k, v, mask)

        # Concatenate heads and apply final linear projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.model_dim)
        output = self.out_proj(attn_output)

        return output, attn_weights

class GroupedQueryAttention(nn.Module):
    """
    Grouped-Query Attention (GQA).
    Query heads are divided into G groups, and each group shares a K and V projection.
    """
    def __init__(self, model_dim: int, num_heads: int, num_kv_heads: int, dropout_p: float = 0.0, bias: bool = False):
        super().__init__()
        if model_dim % num_heads != 0:
            raise ValueError(f"model_dim ({model_dim}) must be divisible by num_heads ({num_heads})")
        if num_heads % num_kv_heads != 0:
            raise ValueError(f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})")

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads # Number of Key/Value groups
        self.head_dim = model_dim // num_heads
        self.num_q_per_kv = self.num_heads // self.num_kv_heads # Number of query heads per KV group

        # Q projection for all heads
        self.q_proj = nn.Linear(model_dim, model_dim, bias=bias)
        # K and V projections for num_kv_heads
        self.k_proj = nn.Linear(model_dim, self.num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(model_dim, self.num_kv_heads * self.head_dim, bias=bias)

        # Output linear projection
        self.out_proj = nn.Linear(model_dim, model_dim, bias=bias)

        self.attention = ScaledDotProductAttention(dropout_p)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        batch_size, seq_len, _ = x.size()

        # Apply Q projection and reshape for multi-head
        # (batch_size, seq_len, model_dim) -> (batch_size, seq_len, num_heads, head_dim)
        # -> (batch_size, num_heads, seq_len, head_dim)
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply K and V projections for num_kv_heads
        # (batch_size, seq_len, model_dim) -> (batch_size, seq_len, num_kv_heads, head_dim)
        # -> (batch_size, num_kv_heads, seq_len, head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Repeat K and V for each query head in its group
        # If num_q_per_kv is 1, no repetition needed (MHA case if num_kv_heads == num_heads)
        # If num_q_per_kv > 1, repeat K/V to match the number of query heads
        if self.num_q_per_kv > 1:
            # (batch_size, num_kv_heads, seq_len, head_dim)
            # -> (batch_size, num_kv_heads, 1, seq_len, head_dim)
            # -> (batch_size, num_kv_heads, num_q_per_kv, seq_len, head_dim)
            # -> (batch_size, num_heads, seq_len, head_dim)
            k = k.unsqueeze(2).expand(batch_size, self.num_kv_heads, self.num_q_per_kv, seq_len, self.head_dim).reshape(batch_size, self.num_heads, seq_len, self.head_dim)
            v = v.unsqueeze(2).expand(batch_size, self.num_kv_heads, self.num_q_per_kv, seq_len, self.head_dim).reshape(batch_size, self.num_heads, seq_len, self.head_dim)

        # Perform scaled dot-product attention
        attn_output, attn_weights = self.attention(q, k, v, mask)

        # Concatenate heads and apply final linear projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.model_dim)
        output = self.out_proj(attn_output)

        return output, attn_weights

# --- Flash Attention (Conceptual Usage) ---
# Flash Attention is not a different mathematical attention mechanism,
# but a highly optimized implementation of scaled dot-product attention
# that reduces memory I/O and speeds up computation on GPUs.
# In PyTorch 2.0+, you can leverage Flash Attention automatically
# by using F.scaled_dot_product_attention.

def demonstrate_flash_attention_concept(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None):
    """
    Demonstrates the conceptual usage of Flash Attention via PyTorch's
    optimized scaled_dot_product_attention.

    Note: The actual performance benefits of Flash Attention are realized
    at the CUDA kernel level and depend on GPU hardware (e.g., NVIDIA A100, H100).
    This function simply shows the API.
    """
    print("\n--- Conceptual Usage of Flash Attention (PyTorch 2.0+ `scaled_dot_product_attention`) ---")
    print("This function leverages highly optimized CUDA kernels for attention computation.")
    print("It provides the same mathematical output as ScaledDotProductAttention but is much faster and more memory-efficient on compatible hardware.")

    # PyTorch's F.scaled_dot_product_attention can automatically use Flash Attention
    # if the inputs are on a CUDA device and the hardware/software supports it.
    # It takes care of the scaling and softmax internally.
    # The 'attn_mask' argument expects a boolean mask (True for attended, False for masked).
    # For causal masking, you'd typically create a lower-triangular mask.
    
    # Create a causal mask for demonstration if not provided
    if mask is None:
        seq_len_q = q.size(-2)
        seq_len_kv = k.size(-2)
        # Create a lower triangular mask
        causal_mask = torch.tril(torch.ones(seq_len_q, seq_len_kv, dtype=torch.bool, device=q.device))
        # Expand for batch and head dimensions if needed
        mask = causal_mask.unsqueeze(0).unsqueeze(0) # (1, 1, seq_len_q, seq_len_kv)

    # Use the optimized function
    # Note: dropout_p is passed directly to the function
    # is_causal=True is a common shortcut for causal masking
    # If using is_causal=True, do NOT pass a separate mask.
    try:
        output_fa = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0) # Mask is handled by attn_mask
        print(f"Flash Attention Output Shape: {output_fa.shape}")
        # You can compare this output with MHA's attention output for numerical equivalence
    except Exception as e:
        print(f"Could not run F.scaled_dot_product_attention. Error: {e}")
        print("This might happen if PyTorch version is < 2.0 or CUDA is not available/configured.")


# --- Test Cases ---
if __name__ == "__main__":
    batch_size = 2
    seq_len = 16
    model_dim = 256
    num_heads_mha = 8
    num_heads_mqa = 8 # MQA will still have 8 query heads, but 1 KV head
    num_heads_gqa = 8
    num_kv_heads_gqa = 2 # GQA with 2 KV groups (8 query heads / 2 KV heads = 4 queries per KV group)
    dropout_rate = 0.1
    use_bias = False # Common in modern LLMs

    # Create dummy input tensor
    input_tensor = torch.randn(batch_size, seq_len, model_dim)

    # Create a causal mask for decoder-only attention
    # This mask ensures a token only attends to itself and previous tokens.
    # Shape: (1, 1, seq_len, seq_len) for broadcasting
    causal_mask = torch.tril(torch.ones(seq_len, seq_len)).view(1, 1, seq_len, seq_len)


    print("="*80)
    print("Demonstrating Multi-Head Attention (MHA)")
    mha_layer = MultiHeadAttention(model_dim, num_heads_mha, dropout_p=dropout_rate, bias=use_bias)
    mha_output, mha_weights = mha_layer(input_tensor, mask=causal_mask)
    print(f"MHA Output Shape: {mha_output.shape}")
    print(f"MHA Attention Weights Shape: {mha_weights.shape}")
    assert mha_output.shape == input_tensor.shape
    assert mha_weights.shape == (batch_size, num_heads_mha, seq_len, seq_len)
    print("MHA test passed.\n")

    print("="*80)
    print("Demonstrating Multi-Query Attention (MQA)")
    mqa_layer = MultiQueryAttention(model_dim, num_heads_mqa, dropout_p=dropout_rate, bias=use_bias)
    mqa_output, mqa_weights = mqa_layer(input_tensor, mask=causal_mask)
    print(f"MQA Output Shape: {mqa_output.shape}")
    print(f"MQA Attention Weights Shape: {mqa_weights.shape}")
    assert mqa_output.shape == input_tensor.shape
    assert mqa_weights.shape == (batch_size, num_heads_mqa, seq_len, seq_len)
    print("MQA test passed.\n")

    print("="*80)
    print("Demonstrating Grouped-Query Attention (GQA)")
    gqa_layer = GroupedQueryAttention(model_dim, num_heads_gqa, num_kv_heads_gqa, dropout_p=dropout_rate, bias=use_bias)
    gqa_output, gqa_weights = gqa_layer(input_tensor, mask=causal_mask)
    print(f"GQA Output Shape: {gqa_output.shape}")
    print(f"GQA Attention Weights Shape: {gqa_weights.shape}")
    assert gqa_output.shape == input_tensor.shape
    assert gqa_weights.shape == (batch_size, num_heads_gqa, seq_len, seq_len)
    print("GQA test passed.\n")

    print("="*80)
    # For Flash Attention, we need to prepare Q, K, V in the correct format (batch, num_heads, seq_len, head_dim)
    # Let's use the Q, K, V from an MHA-like setup for comparison
    mha_q = mha_layer.q_proj(input_tensor).view(batch_size, seq_len, num_heads_mha, mha_layer.head_dim).transpose(1, 2)
    mha_k = mha_layer.k_proj(input_tensor).view(batch_size, seq_len, num_heads_mha, mha_layer.head_dim).transpose(1, 2)
    mha_v = mha_layer.v_proj(input_tensor).view(batch_size, seq_len, num_heads_mha, mha_layer.head_dim).transpose(1, 2)

    # Move to CUDA if available for Flash Attention to potentially activate
    if torch.cuda.is_available():
        print("CUDA is available. Attempting to run Flash Attention on GPU.")
        mha_q = mha_q.cuda()
        mha_k = mha_k.cuda()
        mha_v = mha_v.cuda()
        causal_mask = causal_mask.cuda()
        demonstrate_flash_attention_concept(mha_q, mha_k, mha_v, mask=causal_mask)
    else:
        print("CUDA not available. Flash Attention benefits are primarily on GPU.")
        print("Running conceptual Flash Attention on CPU (will not show speedup).")
        demonstrate_flash_attention_concept(mha_q, mha_k, mha_v, mask=causal_mask)

