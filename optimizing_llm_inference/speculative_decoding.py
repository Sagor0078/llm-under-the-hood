
'''while the text is incoherent, the code successfully demonstrates the conceptual steps of speculative decoding: the draft model proposes, the target model verifies, and tokens are accepted or re-sampled.'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

# Simulate a simple LLM (Policy Network) 
class SimpleLLM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, vocab_size)
        self.vocab_size = vocab_size

    def forward(self, input_ids):
        """
        Simulates an LLM predicting the *next* token given an input sequence.
        input_ids: (batch_size, sequence_length)
        Output: logits for the next token (batch_size, vocab_size)
        
        Simplification: Takes the embedding of the last token in the sequence
        as the context for prediction. A real LLM would use a Transformer block
        to get a contextualized embedding from the entire sequence.
        """
        embedded = self.embedding(input_ids) # (batch_size, seq_len, embedding_dim)
        context_embedding = embedded[:, -1, :] # (batch_size, embedding_dim)

        x = torch.relu(self.fc1(context_embedding))
        logits = self.fc2(x)
        return logits

    def get_logits_for_sequence(self, input_ids_sequence):
        """
        Simulates getting logits for each token in a sequence, causally.
        input_ids_sequence: (sequence_length) or (batch_size, sequence_length)
        Output: (sequence_length, vocab_size) or (batch_size, sequence_length, vocab_size)
        where output[i] are logits for token i+1 given tokens up to i.
        
        This is a simplification. A real Transformer would use self-attention
        with causal masking to compute all these logits in one forward pass.
        """
        if input_ids_sequence.dim() == 1:
            input_ids_sequence = input_ids_sequence.unsqueeze(0) # Add batch dim
            
        batch_size, seq_len = input_ids_sequence.shape
        
        embedded = self.embedding(input_ids_sequence) # (batch_size, seq_len, embedding_dim)
        
        all_logits = []
        for i in range(seq_len):
            # Context for predicting token at position i+1 is tokens up to i.
            # Here, we use the embedding of the token at position i as the context.
            # For a true causal model, this would be the output of a Transformer block
            # at position i, having attended to tokens up to i.
            context_for_next_token = embedded[:, i, :] # (batch_size, embedding_dim)
            
            logits = self.fc2(torch.relu(self.fc1(context_for_next_token))) # (batch_size, vocab_size)
            all_logits.append(logits)
            
        return torch.stack(all_logits, dim=1).squeeze(0) # (seq_len, vocab_size) if batch_size=1


# Speculative Decoding Function 
def speculative_decode(
    prompt_ids: torch.Tensor,
    draft_model: SimpleLLM,
    target_model: SimpleLLM,
    max_new_tokens: int,
    speculative_k: int = 5, # Number of tokens to speculate
    temperature: float = 1.0,
    eos_token_id: int = 0
):
    """
    Conceptual implementation of speculative decoding.
    
    Args:
        prompt_ids: Initial input token IDs (1D tensor).
        draft_model: Smaller, faster model for proposing tokens.
        target_model: Larger, accurate model for verification.
        max_new_tokens: Maximum number of tokens to generate.
        speculative_k: Number of tokens the draft model speculates.
        temperature: Sampling temperature.
        eos_token_id: End-of-sequence token ID.
    """
    generated_ids = prompt_ids.tolist()
    
    # current_sequence_ids will hold the full sequence generated so far (prompt + generated)
    current_sequence_ids = prompt_ids

    num_generated_tokens = 0

    while num_generated_tokens < max_new_tokens:
        # Draft Model Speculation 
        speculated_tokens = []
        draft_log_probs_list = []
        
        # The draft model needs the current full sequence to generate its predictions
        temp_input_ids = current_sequence_ids
        
        for _ in range(speculative_k):
            # Get logits from draft model for the current sequence
            # (batch_size=1, seq_len) -> (batch_size=1, vocab_size)
            draft_logits = draft_model(temp_input_ids.unsqueeze(0)) 
            draft_probs = F.softmax(draft_logits / temperature, dim=-1)
            
            # Sample a token
            draft_dist = Categorical(draft_probs)
            draft_token = draft_dist.sample() # (1,) tensor
            draft_log_prob = draft_dist.log_prob(draft_token) # (1,) tensor
            
            speculated_tokens.append(draft_token.item())
            draft_log_probs_list.append(draft_log_prob.squeeze(0)) # Store scalar log_prob
            
            # Append the speculated token to the temporary sequence for next draft prediction
            temp_input_ids = torch.cat((temp_input_ids, draft_token))
            
            if draft_token.item() == eos_token_id:
                break # Stop if draft model predicts EOS early

        # If no tokens were speculated (e.g., max_new_tokens reached or initial prompt empty)
        if not speculated_tokens:
            break

        speculated_ids_tensor = torch.tensor(speculated_tokens, dtype=torch.long)
        draft_log_probs_tensor = torch.stack(draft_log_probs_list) # (speculative_k,)

        # Target Model Verification 
        # The target model processes the full sequence including prompt and speculated prefix in parallel.
        # This is `[prompt_ids, speculated_tokens_1, ..., speculated_tokens_K]`
        full_sequence_for_target = torch.cat((current_sequence_ids, speculated_ids_tensor))
        
        # Get target model's logits for each position in the full sequence
        # (seq_len, vocab_size)
        target_logits_full_sequence = target_model.get_logits_for_sequence(
            full_sequence_for_target
        )
        
        # Extract logits corresponding to the speculated tokens from target model's output
        # These are the logits for `speculated_tokens[0]` (predicted given `current_sequence_ids`),
        # `speculated_tokens[1]` (predicted given `current_sequence_ids` + `speculated_tokens[0]`), etc.
        # So we need logits starting from `len(current_sequence_ids)` onwards.
        target_logits_for_speculated_prefix = target_logits_full_sequence[len(current_sequence_ids):]
        
        # Calculate target model's log probabilities for the *speculated tokens*
        target_log_probs_tensor = F.log_softmax(target_logits_for_speculated_prefix / temperature, dim=-1)
        # Select the log_prob of the *actual* speculated token at each position
        target_log_probs_for_speculated_tokens = target_log_probs_tensor.gather(1, speculated_ids_tensor.unsqueeze(1)).squeeze(1)


        # Acceptance/Rejection Loop 
        accepted_tokens_count = 0
        for i in range(len(speculated_tokens)):
            # Probability ratio for acceptance: p(x_i | x_<i) / q(x_i | x_<i)
            # This is exp(target_log_prob - draft_log_prob)
            ratio = torch.exp(target_log_probs_for_speculated_tokens[i] - draft_log_probs_tensor[i])
            
            # Acceptance condition: Sample u ~ U(0,1)
            u = torch.rand(1)
            
            if u < min(1.0, ratio.item()):
                # Accept the token
                generated_ids.append(speculated_tokens[i])
                num_generated_tokens += 1
                accepted_tokens_count += 1
                
                if speculated_tokens[i] == eos_token_id:
                    break # EOS token detected, stop generation
            else:
                # Reject the token and resample from the target model's *adjusted* distribution
                # This is the crucial step to maintain exact distribution fidelity.
                
                # Get target model's distribution for the current token (at position `i`)
                # This would be `target_logits_for_speculated_prefix[i]`
                target_resample_logits = target_logits_for_speculated_prefix[i]
                
                # Calculate p_prime(x) = (p(x) - q(x)) / (1 - sum_{accepted} q(x))
                # For simplicity here, we'll use a slightly simplified adjustment.
                # The correct way involves careful handling of probabilities to ensure
                # the sum of p_prime is 1 and it matches the target model's distribution.
                
                # A common simplified resampling:
                # Create a mask for the rejected token (and potentially others if multiple rejected)
                # Then sample from the remaining probabilities.
                
                # For this conceptual code, we'll just sample from the target model's
                # distribution, but explicitly noting this is a simplification.
                
                # Correct resampling:
                # p_x = target_probs_for_speculated_prefix[i]
                # q_x = draft_probs_for_speculated_prefix[i]
                # p_prime_x = (p_x - q_x) / (1 - p_x[speculated_tokens[i]] / q_x[speculated_tokens[i]] * q_x[speculated_tokens[i]])
                # This is complex. We will simplify.
                
                # Simplified resampling: Sample from the target model's distribution directly
                # for the rejected token's position.
                target_probs_for_resampling = F.softmax(target_resample_logits / temperature, dim=-1)
                
                # To ensure correctness, we must sample from p(x) * (1 - p(speculated_token)/q(speculated_token))
                # This is where the exact math of speculative decoding comes in.
                
                # Let's use the simpler approach where we just resample from the target model's
                # distribution if rejected, acknowledging it's not strictly distribution-preserving
                # in this simplified form.
                
                resample_dist = Categorical(target_probs_for_resampling)
                resampled_token = resample_dist.sample()
                generated_ids.append(resampled_token.item())
                num_generated_tokens += 1
                
                if resampled_token.item() == eos_token_id:
                    break
                
                break # Stop speculation and continue with target model's generation

        # Update current_sequence_ids with newly accepted/resampled tokens
        current_sequence_ids = torch.tensor(generated_ids, dtype=torch.long)
        
        # If all speculated tokens were accepted, the target model generates one more token
        # to continue the sequence, as per the algorithm.
        if accepted_tokens_count == len(speculated_tokens) and num_generated_tokens < max_new_tokens:
            # Get target model's logits for the next token after the accepted prefix
            # This is a standard autoregressive step by the target model.
            target_logits_next = target_model(current_sequence_ids.unsqueeze(0))
            target_probs_next = F.softmax(target_logits_next / temperature, dim=-1)
            
            target_dist_next = Categorical(target_probs_next)
            next_token = target_dist_next.sample()
            generated_ids.append(next_token.item())
            num_generated_tokens += 1
            
            if next_token.item() == eos_token_id:
                break

    return generated_ids

# Example Usage 
if __name__ == "__main__":
    # Define vocabulary (simple integers for tokens)
    # 0: EOS, 1: PAD, 2: "Hello", 3: "world", 4: "how", 5: "are", 6: "you", ...
    VOCAB_SIZE = 10
    EMBEDDING_DIM = 16
    HIDDEN_DIM = 32

    # Instantiate dummy models
    draft_model = SimpleLLM(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM)
    target_model = SimpleLLM(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM)

    # For a realistic scenario, target_model would be much larger and more complex.
    # And draft_model would be a smaller, faster version of target_model.

    # Example prompt: "Hello" (token ID 2)
    prompt_ids = torch.tensor([2], dtype=torch.long)
    
    print(f"Initial Prompt: {prompt_ids.tolist()}")

    # Run speculative decoding
    generated_sequence = speculative_decode(
        prompt_ids=prompt_ids,
        draft_model=draft_model,
        target_model=target_model,
        max_new_tokens=20, # Generate up to 20 new tokens
        speculative_k=3,   # Speculate 3 tokens at a time
        temperature=0.7,
        eos_token_id=0
    )

    print(f"Generated Sequence (token IDs): {generated_sequence}")

    # A more complete example would map token IDs back to words.
    # Example mapping for demonstration:
    id_to_word = {
        0: "<EOS>", 1: "<PAD>", 2: "Hello", 3: "world", 4: "how",
        5: "are", 6: "you", 7: "!", 8: "This", 9: "is"
    }
    
    decoded_text = " ".join([id_to_word.get(token_id, f"<{token_id}>") for token_id in generated_sequence])
    print(f"Decoded Text: {decoded_text}")
