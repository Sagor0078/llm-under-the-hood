import torch
import torch.nn.functional as F
from typing import Optional, Tuple, List
import numpy as np

class SpeculativeDecoder:
    """
    Speculative Decoding implementation for accelerating autoregressive generation.
    
    This implementation uses a smaller draft model to propose tokens and a larger
    target model to verify them, maintaining the target model's output distribution.
    """
    
    def __init__(self, draft_model, target_model, device='cuda'):
        """
        Initialize the speculative decoder.
        
        Args:
            draft_model: Smaller, faster model for generating proposals
            target_model: Larger, target model for verification
            device: Device to run models on
        """
        self.draft_model = draft_model
        self.target_model = target_model
        self.device = device
        
        # Move models to device and set to eval mode
        self.draft_model.to(device).eval()
        self.target_model.to(device).eval()
    
    def sample_token(self, logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """Sample a token from logits with temperature scaling."""
        if temperature == 0:
            return torch.argmax(logits, dim=-1)
        
        probs = F.softmax(logits / temperature, dim=-1)
        return torch.multinomial(probs, 1).squeeze(-1)
    
    def compute_acceptance_prob(self, 
                              target_logits: torch.Tensor, 
                              draft_logits: torch.Tensor, 
                              proposed_tokens: torch.Tensor) -> torch.Tensor:
        """
        Compute acceptance probabilities for proposed tokens.
        
        Args:
            target_logits: Logits from target model [batch_size, vocab_size]
            draft_logits: Logits from draft model [batch_size, vocab_size]
            proposed_tokens: Proposed token indices [batch_size]
        
        Returns:
            Acceptance probabilities [batch_size]
        """
        target_probs = F.softmax(target_logits, dim=-1)
        draft_probs = F.softmax(draft_logits, dim=-1)
        
        # Get probabilities for proposed tokens
        target_token_probs = target_probs.gather(-1, proposed_tokens.unsqueeze(-1)).squeeze(-1)
        draft_token_probs = draft_probs.gather(-1, proposed_tokens.unsqueeze(-1)).squeeze(-1)
        
        # Compute acceptance probability: min(1, p_target / p_draft)
        acceptance_probs = torch.min(
            torch.ones_like(target_token_probs),
            target_token_probs / (draft_token_probs + 1e-10)  # Add epsilon for numerical stability
        )
        
        return acceptance_probs
    
    def generate_draft_sequence(self, 
                              input_ids: torch.Tensor, 
                              num_speculative_tokens: int = 4,
                              temperature: float = 1.0) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Generate speculative tokens using the draft model.
        
        Args:
            input_ids: Input token sequence [batch_size, seq_len]
            num_speculative_tokens: Number of tokens to generate speculatively
            temperature: Sampling temperature
        
        Returns:
            proposed_sequence: Extended sequence with proposed tokens
            draft_logits_list: List of logits for each proposed token
        """
        batch_size = input_ids.shape[0]
        current_input = input_ids
        draft_logits_list = []
        
        with torch.no_grad():
            for _ in range(num_speculative_tokens):
                # Get logits from draft model
                outputs = self.draft_model(current_input)
                logits = outputs.logits[:, -1, :]  # Last token logits
                draft_logits_list.append(logits)
                
                # Sample next token
                next_token = self.sample_token(logits, temperature)
                current_input = torch.cat([current_input, next_token.unsqueeze(1)], dim=1)
        
        return current_input, draft_logits_list
    
    def verify_with_target_model(self, 
                                input_ids: torch.Tensor,
                                proposed_sequence: torch.Tensor,
                                draft_logits_list: List[torch.Tensor],
                                temperature: float = 1.0) -> Tuple[torch.Tensor, int]:
        """
        Verify proposed tokens using the target model.
        
        Args:
            input_ids: Original input sequence
            proposed_sequence: Sequence with proposed tokens
            draft_logits_list: Logits from draft model for each proposed token
            temperature: Sampling temperature
        
        Returns:
            accepted_sequence: Sequence with accepted tokens
            num_accepted: Number of accepted tokens
        """
        batch_size = input_ids.shape[0]
        original_len = input_ids.shape[1]
        num_proposed = len(draft_logits_list)
        
        with torch.no_grad():
            # Get target model logits for the entire proposed sequence
            target_outputs = self.target_model(proposed_sequence)
            target_logits = target_outputs.logits
            
            accepted_sequence = input_ids.clone()
            num_accepted = 0
            
            # Check each proposed token
            for i in range(num_proposed):
                pos = original_len + i - 1  # Position in target_logits
                proposed_token = proposed_sequence[:, original_len + i]
                
                # Get logits at the position before the proposed token
                current_target_logits = target_logits[:, pos, :]
                current_draft_logits = draft_logits_list[i]
                
                # Compute acceptance probability
                accept_probs = self.compute_acceptance_prob(
                    current_target_logits, current_draft_logits, proposed_token
                )
                
                # Sample acceptance decision
                random_vals = torch.rand_like(accept_probs)
                accepted = random_vals < accept_probs
                
                if accepted.all():
                    # Accept the token
                    accepted_sequence = torch.cat([
                        accepted_sequence, 
                        proposed_token.unsqueeze(1)
                    ], dim=1)
                    num_accepted += 1
                else:
                    # Reject the token and resample from target distribution
                    # Create modified distribution for rejection sampling
                    target_probs = F.softmax(current_target_logits / temperature, dim=-1)
                    draft_probs = F.softmax(current_draft_logits / temperature, dim=-1)
                    
                    # Compute modified probabilities
                    modified_probs = torch.clamp(target_probs - draft_probs, min=0)
                    modified_probs = modified_probs / (modified_probs.sum(dim=-1, keepdim=True) + 1e-10)
                    
                    # Sample from modified distribution
                    new_token = torch.multinomial(modified_probs, 1).squeeze(-1)
                    accepted_sequence = torch.cat([
                        accepted_sequence, 
                        new_token.unsqueeze(1)
                    ], dim=1)
                    num_accepted += 1
                    break  # Stop after first rejection
        
        return accepted_sequence, num_accepted
    
    def generate(self, 
                input_ids: torch.Tensor,
                max_length: int = 100,
                num_speculative_tokens: int = 4,
                temperature: float = 1.0,
                pad_token_id: Optional[int] = None) -> torch.Tensor:
        """
        Generate text using speculative decoding.
        
        Args:
            input_ids: Input token sequence [batch_size, seq_len]
            max_length: Maximum total sequence length
            num_speculative_tokens: Number of tokens to propose per iteration
            temperature: Sampling temperature
            pad_token_id: Padding token ID
        
        Returns:
            Generated sequence [batch_size, seq_len]
        """
        current_sequence = input_ids
        
        while current_sequence.shape[1] < max_length:
            # Generate speculative tokens
            proposed_sequence, draft_logits = self.generate_draft_sequence(
                current_sequence, num_speculative_tokens, temperature
            )
            
            # Verify with target model
            accepted_sequence, num_accepted = self.verify_with_target_model(
                current_sequence, proposed_sequence, draft_logits, temperature
            )
            
            current_sequence = accepted_sequence
            
            # Break if we've reached max length
            if current_sequence.shape[1] >= max_length:
                current_sequence = current_sequence[:, :max_length]
                break
        
        return current_sequence


# Example usage and benchmarking functions
class MockLanguageModel(torch.nn.Module):
    """Mock language model for demonstration purposes."""
    
    def __init__(self, vocab_size: int = 32000, hidden_size: int = 768):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, hidden_size)
        self.transformer = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(hidden_size, 8, batch_first=True),
            num_layers=6
        )
        self.lm_head = torch.nn.Linear(hidden_size, vocab_size)
        self.vocab_size = vocab_size
    
    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = self.transformer(x)
        logits = self.lm_head(x)
        
        # Return object with logits attribute to match transformers interface
        class Output:
            def __init__(self, logits):
                self.logits = logits
        
        return Output(logits)


def benchmark_speculative_decoding():
    """Benchmark speculative decoding vs regular generation."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create mock models (draft model is smaller/faster)
    draft_model = MockLanguageModel(vocab_size=1000, hidden_size=256)
    target_model = MockLanguageModel(vocab_size=1000, hidden_size=768)
    
    # Initialize speculative decoder
    decoder = SpeculativeDecoder(draft_model, target_model, device)
    
    # Create sample input
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, 1000, (batch_size, seq_len)).to(device)
    
    print("Running speculative decoding example...")
    
    # Generate using speculative decoding
    import time
    start_time = time.time()
    generated_sequence = decoder.generate(
        input_ids, 
        max_length=50, 
        num_speculative_tokens=4,
        temperature=1.0
    )
    spec_time = time.time() - start_time
    
    print(f"Speculative decoding completed in {spec_time:.3f}s")
    print(f"Generated sequence shape: {generated_sequence.shape}")
    print(f"Input length: {input_ids.shape[1]}, Output length: {generated_sequence.shape[1]}")
    
    return generated_sequence


# Additional utility functions
def calculate_acceptance_rate(num_accepted_list: List[int], 
                            num_proposed_per_iteration: int) -> float:
    """Calculate the average acceptance rate across iterations."""
    if not num_accepted_list:
        return 0.0
    
    total_accepted = sum(num_accepted_list)
    total_proposed = len(num_accepted_list) * num_proposed_per_iteration
    return total_accepted / total_proposed if total_proposed > 0 else 0.0


def estimate_speedup(acceptance_rate: float, 
                    draft_speed_ratio: float = 3.0) -> float:
    """
    Estimate theoretical speedup from speculative decoding.
    
    Args:
        acceptance_rate: Average rate of token acceptance
        draft_speed_ratio: How much faster the draft model is than target model
    
    Returns:
        Estimated speedup factor
    """
    # Simplified speedup model
    # Speedup ≈ 1 + (acceptance_rate * num_speculative_tokens * draft_speed_advantage)
    return 1 + (acceptance_rate * 4 * (draft_speed_ratio - 1) / draft_speed_ratio)


if __name__ == "__main__":
    # Run the benchmark
    result = benchmark_speculative_decoding()
    print("Benchmark completed successfully!")