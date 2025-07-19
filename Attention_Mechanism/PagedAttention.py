import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Dict, Tuple, Optional

# This simulates the global memory pool and block size
KV_CACHE_BLOCK_SIZE = 16  # Number of tokens per KV cache block
MAX_PHYSICAL_BLOCKS = 1024 # Total number of physical blocks available in GPU memory
HEAD_DIM = 128            # Dimension of each attention head
N_KV_HEADS = 4            # Number of Key/Value heads (Adjusted to match example usage)

# Simulate the global pool of physical KV cache blocks
# Each block stores (N_KV_HEADS, KV_CACHE_BLOCK_SIZE, HEAD_DIM) for Keys
# and (N_KV_HEADS, KV_CACHE_BLOCK_SIZE, HEAD_DIM) for Values
# For simplicity, we'll store them as a single tensor:
# (MAX_PHYSICAL_BLOCKS, N_KV_HEADS, KV_CACHE_BLOCK_SIZE, HEAD_DIM)
global_kv_cache_pool = {
    "keys": torch.zeros(MAX_PHYSICAL_BLOCKS, N_KV_HEADS, KV_CACHE_BLOCK_SIZE, HEAD_DIM, dtype=torch.float16),
    "values": torch.zeros(MAX_PHYSICAL_BLOCKS, N_KV_HEADS, KV_CACHE_BLOCK_SIZE, HEAD_DIM, dtype=torch.float16)
}

# Simulate a simple free list for physical blocks
free_physical_block_ids = list(range(MAX_PHYSICAL_BLOCKS))


class PagedAttentionKVManager:
    """
    Conceptual KV cache manager for PagedAttention.
    Manages the mapping from logical token positions in a sequence
    to physical block IDs in the global KV cache pool.
    """
    def __init__(self):
        # Stores block tables for each active sequence
        # {sequence_id: List[physical_block_id]}
        self.sequence_block_tables: Dict[int, List[int]] = {}
        # Tracks the number of filled tokens in the last block of each sequence
        # {sequence_id: int}
        self.sequence_last_block_filled_count: Dict[int, int] = {}
        self.next_sequence_id = 0

    def _allocate_physical_block(self) -> int:
        """Allocates a new physical block from the global pool."""
        if not free_physical_block_ids:
            raise RuntimeError("Out of physical KV cache blocks!")
        block_id = free_physical_block_ids.pop(0)
        # In a real system, you might clear this block or manage reference counts
        # For simplicity, we just return the ID.
        return block_id

    def _free_physical_block(self, block_id: int):
        """Frees a physical block, returning it to the global pool."""
        free_physical_block_ids.append(block_id)
        # In a real system, you might zero out the block for security/cleanliness
        # or manage reference counts if blocks are shared.

    def create_sequence(self) -> int:
        """Creates a new sequence and initializes its block table."""
        seq_id = self.next_sequence_id
        self.next_sequence_id += 1
        self.sequence_block_tables[seq_id] = []
        self.sequence_last_block_filled_count[seq_id] = 0
        print(f"KV Manager: Created new sequence with ID: {seq_id}")
        return seq_id

    def release_sequence(self, seq_id: int):
        """Releases a sequence and frees all its associated physical blocks."""
        if seq_id in self.sequence_block_tables:
            for block_id in self.sequence_block_tables[seq_id]:
                self._free_physical_block(block_id)
            del self.sequence_block_tables[seq_id]
            del self.sequence_last_block_filled_count[seq_id]
            print(f"KV Manager: Released sequence {seq_id} and freed its blocks.")
        else:
            print(f"KV Manager: Sequence {seq_id} not found.")

    def append_kv_to_cache(
        self,
        seq_id: int,
        key_tensor: torch.Tensor,   # (1, N_KV_HEADS, HEAD_DIM) for a single token
        value_tensor: torch.Tensor, # (1, N_KV_HEADS, HEAD_DIM) for a single token
        token_index_in_sequence: int # Absolute index of the token in its logical sequence
    ):
        """
        Appends a single token's KV to the cache for a specific sequence.
        Handles block allocation and writing to the correct physical location.
        """
        if seq_id not in self.sequence_block_tables:
            raise ValueError(f"Sequence ID {seq_id} not found. Create it first.")

        current_block_table = self.sequence_block_tables[seq_id]
        current_filled_count = self.sequence_last_block_filled_count[seq_id]

        logical_block_idx = token_index_in_sequence // KV_CACHE_BLOCK_SIZE
        offset_in_block = token_index_in_sequence % KV_CACHE_BLOCK_SIZE

        # Check if we need to allocate a new block
        if logical_block_idx >= len(current_block_table):
            new_physical_block_id = self._allocate_physical_block()
            current_block_table.append(new_physical_block_id)
            self.sequence_last_block_filled_count[seq_id] = 0 # Reset count for new block
            print(f"KV Manager: Seq {seq_id} - Allocated new physical block {new_physical_block_id} for logical block {logical_block_idx}")

        physical_block_id = current_block_table[logical_block_idx]

        # Write KV to the global pool at the correct physical location and offset
        # key_tensor and value_tensor are (1, N_KV_HEADS, HEAD_DIM)
        # We need to squeeze the first dimension (seq_len=1)
        global_kv_cache_pool["keys"][physical_block_id, :, offset_in_block, :] = key_tensor.squeeze(0)
        global_kv_cache_pool["values"][physical_block_id, :, offset_in_block, :] = value_tensor.squeeze(0)

        # Update filled count for the last block
        if logical_block_idx == len(current_block_table) - 1:
            self.sequence_last_block_filled_count[seq_id] = offset_in_block + 1

        # print(f"KV Manager: Seq {seq_id} - Stored token {token_index_in_sequence} in physical block {physical_block_id} at offset {offset_in_block}")

    def get_kv_from_cache(
        self,
        seq_id: int,
        sequence_length: int # Total length of the sequence to retrieve
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves the full KV cache for a given sequence by gathering from non-contiguous blocks.
        Returns: (keys_tensor, values_tensor) of shape (1, sequence_length, N_KV_HEADS, HEAD_DIM)
        """
        if seq_id not in self.sequence_block_tables:
            raise ValueError(f"Sequence ID {seq_id} not found.")

        current_block_table = self.sequence_block_tables[seq_id]

        # Calculate how many logical blocks are needed for the current sequence length
        num_logical_blocks = (sequence_length + KV_CACHE_BLOCK_SIZE - 1) // KV_CACHE_BLOCK_SIZE

        if num_logical_blocks > len(current_block_table):
            raise ValueError(f"Requested sequence length {sequence_length} requires {num_logical_blocks} blocks, "
                             f"but only {len(current_block_table)} are allocated for sequence {seq_id}.")

        retrieved_keys_list = []
        retrieved_values_list = []

        for i in range(num_logical_blocks):
            physical_block_id = current_block_table[i]

            # Determine how many tokens to retrieve from this block
            tokens_in_this_block = KV_CACHE_BLOCK_SIZE
            if i == num_logical_blocks - 1: # Last block might not be full
                tokens_in_this_block = sequence_length % KV_CACHE_BLOCK_SIZE
                if tokens_in_this_block == 0 and sequence_length > 0: # If sequence_length is a multiple of block size
                    tokens_in_this_block = KV_CACHE_BLOCK_SIZE
                elif sequence_length == 0: # Handle empty sequence case
                    tokens_in_this_block = 0

            if tokens_in_this_block > 0:
                # Retrieve the actual data from the global pool
                # Shape: (N_KV_HEADS, tokens_in_this_block, HEAD_DIM)
                retrieved_keys_list.append(global_kv_cache_pool["keys"][physical_block_id, :, :tokens_in_this_block, :])
                retrieved_values_list.append(global_kv_cache_pool["values"][physical_block_id, :, :tokens_in_this_block, :])

        if not retrieved_keys_list: # Handle case of empty sequence
            return (
                torch.empty(1, 0, N_KV_HEADS, HEAD_DIM, dtype=torch.float16),
                torch.empty(1, 0, N_KV_HEADS, HEAD_DIM, dtype=torch.float16)
            )

        # Concatenate retrieved blocks
        # Keys shape: (N_KV_HEADS, sequence_length, HEAD_DIM)
        # Values shape: (N_KV_HEADS, sequence_length, HEAD_DIM)
        all_keys = torch.cat(retrieved_keys_list, dim=1)
        all_values = torch.cat(retrieved_values_list, dim=1)

        # Transpose to (1, sequence_length, N_KV_HEADS, HEAD_DIM) to match attention input
        all_keys = all_keys.permute(1, 0, 2).unsqueeze(0)
        all_values = all_values.permute(1, 0, 2).unsqueeze(0)

        # print(f"KV Manager: Seq {seq_id} - Retrieved KV cache of length {sequence_length}. Keys shape: {all_keys.shape}")
        return all_keys, all_values


class PagedAttention(nn.Module):
    """
    Conceptual PagedAttention module.
    This module would replace the standard MultiHeadAttentionBlock in a Transformer,
    interacting with the PagedAttentionKVManager.
    """
    def __init__(self, model_dim: int, n_heads: int, n_kv_heads: Optional[int] = None):
        super().__init__()
        self.n_q_heads = n_heads
        self.n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        self.head_dim = model_dim // self.n_q_heads # Calculate head_dim based on query heads

        self.n_repeats = self.n_q_heads // self.n_kv_heads

        assert self.head_dim == HEAD_DIM, "Head dimension mismatch with global config."
        assert self.n_kv_heads == N_KV_HEADS, "N_KV_HEADS mismatch with global config."

        self.wq = nn.Linear(model_dim, self.n_q_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(model_dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(model_dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_q_heads * self.head_dim, model_dim, bias=False)

        # Cast linear layer weights to float16 to match input tensor dtype
        self.wq.half()
        self.wk.half()
        self.wv.half()
        self.wo.half()


        # In a real LLM serving system, this manager would be a singleton or
        # passed globally. For this example, we instantiate it here.
        self.kv_manager = PagedAttentionKVManager()

    @staticmethod
    def _repeat_kv(x: torch.Tensor, n_repeats: int) -> torch.Tensor:
        """Helper to repeat KV heads for GQA."""
        batch_size, seq_len, n_kv_heads, head_dim = x.shape
        if n_repeats == 1:
            return x
        return (
            x[:, :, :, None, :]
            .expand(batch_size, seq_len, n_kv_heads, n_repeats, head_dim)
            .reshape(batch_size, seq_len, n_kv_heads * n_repeats, head_dim)
        )

    @staticmethod
    def _scaled_dot_product_attention(query, key, value, head_dim):
        """Standard scaled dot product attention."""
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(head_dim)
        attention_weights = F.softmax(attention_scores, dim=-1)
        output = attention_weights @ value
        return output

    def forward(
        self,
        x: torch.Tensor,  # (batch_size, 1, model_dim) - current token embedding
        seq_id: int,      # ID of the sequence this token belongs to
        token_index_in_sequence: int, # Absolute index of the token in its logical sequence
        # freqs_complex: torch.Tensor # RoPE frequencies would be passed here in a real model
    ):
        batch_size, seq_len, model_dim = x.shape
        assert seq_len == 1, "PagedAttention processes one token at a time."
        assert batch_size == 1, "This conceptual implementation handles one sequence at a time."

        # 1. Project Query, Key, Value for the current token
        query = self.wq(x).view(batch_size, seq_len, self.n_q_heads, self.head_dim)
        key = self.wk(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        value = self.wv(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        # (Optional) Apply Rotary Positional Embeddings here to query and key
        # query = apply_rotary_embeddings(query, freqs_complex)
        # key = apply_rotary_embeddings(key, freqs_complex)

        # 2. Store the current token's KV to the Paged KV Cache
        self.kv_manager.append_kv_to_cache(
            seq_id,
            key.squeeze(0), # Pass (N_KV_HEADS, HEAD_DIM) for single token
            value.squeeze(0), # Pass (N_KV_HEADS, HEAD_DIM) for single token
            token_index_in_sequence
        )

        # 3. Retrieve the full KV history for this sequence from the Paged KV Cache
        # This will gather K/V from potentially non-contiguous physical blocks
        # Shapes: (1, current_sequence_length, N_KV_HEADS, HEAD_DIM)
        current_sequence_length = token_index_in_sequence + 1
        cached_keys, cached_values = self.kv_manager.get_kv_from_cache(seq_id, current_sequence_length)

        # 4. Apply repeat_kv for GQA
        keys_expanded = self._repeat_kv(cached_keys, self.n_repeats)
        values_expanded = self._repeat_kv(cached_values, self.n_repeats)

        # 5. Transpose for attention computation
        query = query.transpose(1, 2) # (B, H_Q, 1, Head_Dim)
        keys_expanded = keys_expanded.transpose(1, 2) # (B, H_Q, current_seq_len, Head_Dim)
        values_expanded = values_expanded.transpose(1, 2) # (B, H_Q, current_seq_len, Head_Dim)

        # 6. Perform attention
        output = self._scaled_dot_product_attention(
            query, keys_expanded, values_expanded, self.head_dim
        ) # (B, H_Q, 1, Head_Dim)

        # 7. Reshape and final projection
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, model_dim)
        return self.wo(output)


# --- Example Usage ---
if __name__ == "__main__":
    MODEL_DIM = 512
    # Adjusted N_HEADS to match global HEAD_DIM calculation
    N_HEADS = 4 # Example for GQA, 512 / 4 = 128
    N_KV_HEADS_GQA = 4 # Example for GQA, now matches N_HEADS

    print(f"--- PagedAttention Conceptual Demo ---")
    print(f"KV Cache Block Size: {KV_CACHE_BLOCK_SIZE}")
    print(f"Max Physical Blocks: {MAX_PHYSICAL_BLOCKS}")
    print(f"Model Dim: {MODEL_DIM}, N_Heads: {N_HEADS}, N_KV_Heads: {N_KV_HEADS_GQA}")

    # Instantiate the PagedAttention module
    paged_attention_layer = PagedAttention(MODEL_DIM, N_HEADS, N_KV_HEADS_GQA)
    kv_manager = paged_attention_layer.kv_manager # Access the manager directly for demo

    # Simulate processing a sequence token by token
    input_sequence_length = 25 # Example sequence length

    # Create a new sequence
    seq_id_1 = kv_manager.create_sequence()

    print(f"\nProcessing sequence {seq_id_1} with {input_sequence_length} tokens...")
    for i in range(input_sequence_length):
        # Simulate input embedding for the current token
        current_token_embedding = torch.randn(1, 1, MODEL_DIM, dtype=torch.float16)

        # Forward pass through PagedAttention
        # In a real Transformer, this would be part of a larger EncoderBlock
        output = paged_attention_layer(current_token_embedding, seq_id_1, i)
        # print(f"Token {i}: Output shape {output.shape}")

        # Verify block allocation
        logical_block_idx = i // KV_CACHE_BLOCK_SIZE
        offset_in_block = i % KV_CACHE_BLOCK_SIZE

        # Check if a new block was allocated for this token
        if offset_in_block == 0 and i > 0:
            print(f"  --> New logical block {logical_block_idx} allocated for token {i}.")

        # Verify the KV was stored in the correct physical block
        physical_block_id = kv_manager.sequence_block_tables[seq_id_1][logical_block_idx]
        # print(f"  Token {i} stored in physical block {physical_block_id} at offset {offset_in_block}")

    print(f"\nFinished processing sequence {seq_id_1}.")
    print(f"Total logical blocks used for sequence {seq_id_1}: {len(kv_manager.sequence_block_tables[seq_id_1])}")
    print(f"Last block filled count: {kv_manager.sequence_last_block_filled_count[seq_id_1]}")
    print(f"Free physical blocks remaining: {len(free_physical_block_ids)}")

    # Simulate another sequence to show shared global pool
    seq_id_2 = kv_manager.create_sequence()
    print(f"\nProcessing sequence {seq_id_2} with 5 tokens...")
    for i in range(5):
        current_token_embedding = torch.randn(1, 1, MODEL_DIM, dtype=torch.float16)
        output = paged_attention_layer(current_token_embedding, seq_id_2, i)
    print(f"Finished processing sequence {seq_id_2}.")
    print(f"Total logical blocks used for sequence {seq_id_2}: {len(kv_manager.sequence_block_tables[seq_id_2])}")
    print(f"Free physical blocks remaining: {len(free_physical_block_ids)}")

    # Release a sequence and observe blocks being freed
    print(f"\nReleasing sequence {seq_id_1}...")
    kv_manager.release_sequence(seq_id_1)
    print(f"Free physical blocks after release: {len(free_physical_block_ids)}")

    # Try to access a released sequence (should raise error)
    print("\nAttempting to access a released sequence (expected error):")
    try:
        kv_manager.get_kv_from_cache(seq_id_1, 10)
    except ValueError as e:
        print(f"Caught expected error: {e}")

    # Demonstrate retrieving KV cache for a sequence
    print(f"\nRetrieving full KV cache for sequence {seq_id_2} (length 5):")
    retrieved_keys, retrieved_values = kv_manager.get_kv_from_cache(seq_id_2, 5)
    print(f"Retrieved keys shape: {retrieved_keys.shape}")
    print(f"Retrieved values shape: {retrieved_values.shape}")
    assert retrieved_keys.shape == (1, 5, N_KV_HEADS, HEAD_DIM)
    assert retrieved_values.shape == (1, 5, N_KV_HEADS, HEAD_DIM)
    print("KV cache retrieval successful.")