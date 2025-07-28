import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
import numpy as np

class GSPOTrainer:
    """
    Group Sequence Policy Optimization (GSPO) implementation in PyTorch.
    
    Based on the paper: "Group Sequence Policy Optimization" by Zheng et al.
    """
    
    def __init__(
        self,
        policy_model: nn.Module,
        old_policy_model: nn.Module,
        clip_range: float = 0.02,  # Note: GSPO typically uses smaller clipping ranges
        learning_rate: float = 1e-5,
        optimizer_cls = torch.optim.AdamW,
        device: str = "cuda"
    ):
        """
        Initialize GSPO trainer.
        
        Args:
            policy_model: Current policy model to be optimized
            old_policy_model: Reference policy model (typically frozen)
            clip_range: Clipping range for importance ratios
            learning_rate: Learning rate for optimization
            optimizer_cls: Optimizer class
            device: Device to run on
        """
        self.policy_model = policy_model
        self.old_policy_model = old_policy_model
        self.clip_range = clip_range
        self.device = device
        
        # Freeze old policy model
        for param in self.old_policy_model.parameters():
            param.requires_grad = False
        
        # Initialize optimizer
        self.optimizer = optimizer_cls(
            self.policy_model.parameters(), 
            lr=learning_rate
        )
        
    def compute_sequence_likelihood(
        self, 
        model: nn.Module, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor,
        response_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute sequence likelihood for responses.
        
        Args:
            model: Language model
            input_ids: Input token ids [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]  
            response_mask: Mask indicating response tokens [batch_size, seq_len]
            
        Returns:
            log_probs: Log probabilities for each sequence [batch_size]
        """
        with torch.no_grad() if model == self.old_policy_model else torch.enable_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # [batch_size, seq_len, vocab_size]
            
            # Shift for next token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            shift_response_mask = response_mask[..., 1:].contiguous()
            
            # Compute log probabilities
            log_probs = F.log_softmax(shift_logits, dim=-1)
            
            # Gather log probabilities for actual tokens
            token_log_probs = log_probs.gather(
                dim=-1, 
                index=shift_labels.unsqueeze(-1)
            ).squeeze(-1)  # [batch_size, seq_len-1]
            
            # Mask out non-response tokens and sum over sequence
            masked_log_probs = token_log_probs * shift_response_mask
            sequence_log_probs = masked_log_probs.sum(dim=-1)  # [batch_size]
            
            # Normalize by response length
            response_lengths = shift_response_mask.sum(dim=-1).clamp(min=1)
            normalized_log_probs = sequence_log_probs / response_lengths
            
        return normalized_log_probs
    
    def compute_importance_ratios(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        response_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute sequence-level importance ratios s_i(θ).
        
        Args:
            input_ids: Input token ids [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            response_mask: Mask for response tokens [batch_size, seq_len]
            
        Returns:
            importance_ratios: s_i(θ) for each sequence [batch_size]
        """
        # Compute log probabilities under both policies
        current_log_probs = self.compute_sequence_likelihood(
            self.policy_model, input_ids, attention_mask, response_mask
        )
        old_log_probs = self.compute_sequence_likelihood(
            self.old_policy_model, input_ids, attention_mask, response_mask
        )
        
        # Compute importance ratios: (π_θ(y|x) / π_θ_old(y|x))^(1/|y|)
        log_ratio = current_log_probs - old_log_probs
        importance_ratios = torch.exp(log_ratio)
        
        return importance_ratios
    
    def compute_group_advantages(
        self, 
        rewards: torch.Tensor, 
        group_size: int
    ) -> torch.Tensor:
        """
        Compute group-based advantages.
        
        Args:
            rewards: Rewards for each response [batch_size]
            group_size: Number of responses per query
            
        Returns:
            advantages: Normalized advantages [batch_size]
        """
        batch_size = rewards.size(0)
        num_groups = batch_size // group_size
        
        # Reshape to [num_groups, group_size]
        rewards_grouped = rewards.view(num_groups, group_size)
        
        # Compute group statistics
        group_means = rewards_grouped.mean(dim=1, keepdim=True)
        group_stds = rewards_grouped.std(dim=1, keepdim=True).clamp(min=1e-8)
        
        # Normalize advantages
        advantages_grouped = (rewards_grouped - group_means) / group_stds
        
        # Reshape back to [batch_size]
        advantages = advantages_grouped.view(batch_size)
        
        return advantages
    
    def gspo_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        response_mask: torch.Tensor,
        rewards: torch.Tensor,
        group_size: int
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute GSPO loss.
        
        Args:
            input_ids: Input token ids [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            response_mask: Response token mask [batch_size, seq_len]
            rewards: Rewards for each response [batch_size]
            group_size: Number of responses per query group
            
        Returns:
            loss: GSPO loss
            info: Dictionary with training statistics
        """
        # Compute importance ratios
        importance_ratios = self.compute_importance_ratios(
            input_ids, attention_mask, response_mask
        )
        
        # Compute advantages
        advantages = self.compute_group_advantages(rewards, group_size)
        
        # Apply clipping
        clipped_ratios = torch.clamp(
            importance_ratios,
            1 - self.clip_range,
            1 + self.clip_range
        )
        
        # Compute objectives
        obj1 = importance_ratios * advantages
        obj2 = clipped_ratios * advantages
        
        # Take minimum (PPO-style clipping)
        clipped_obj = torch.min(obj1, obj2)
        
        # Average over groups
        batch_size = input_ids.size(0)
        num_groups = batch_size // group_size
        
        grouped_obj = clipped_obj.view(num_groups, group_size)
        group_avg_obj = grouped_obj.mean(dim=1)
        
        # Final loss (negative because we want to maximize)
        loss = -group_avg_obj.mean()
        
        # Compute statistics
        with torch.no_grad():
            clipping_fraction = (importance_ratios != clipped_ratios).float().mean()
            mean_importance_ratio = importance_ratios.mean()
            mean_advantage = advantages.mean()
            mean_reward = rewards.mean()
        
        info = {
            'loss': loss.item(),
            'clipping_fraction': clipping_fraction.item(),
            'mean_importance_ratio': mean_importance_ratio.item(),
            'mean_advantage': mean_advantage.item(),
            'mean_reward': mean_reward.item(),
        }
        
        return loss, info
    
    def train_step(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        response_mask: torch.Tensor,
        rewards: torch.Tensor,
        group_size: int
    ) -> Dict:
        """
        Perform one training step.
        """
        self.optimizer.zero_grad()
        
        loss, info = self.gspo_loss(
            input_ids, attention_mask, response_mask, rewards, group_size
        )
        
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        return info


class GSPOTokenTrainer(GSPOTrainer):
    """
    GSPO-token variant that allows token-level advantage customization.
    """
    
    def compute_token_importance_ratios(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        response_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute token-level importance ratios for GSPO-token.
        
        Returns:
            sequence_ratios: Sequence-level ratios [batch_size]
            token_ratios: Token-level ratios [batch_size, seq_len]
        """
        # Get sequence-level ratios
        sequence_ratios = self.compute_importance_ratios(
            input_ids, attention_mask, response_mask
        )
        
        # Compute token-level likelihood ratios
        current_log_probs_seq = self.compute_sequence_likelihood(
            self.policy_model, input_ids, attention_mask, response_mask
        )
        
        with torch.no_grad():
            # Get token-level probabilities from current policy
            outputs = self.policy_model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            
            token_probs = F.softmax(logits, dim=-1)
            current_token_probs = token_probs.gather(
                dim=-1, index=shift_labels.unsqueeze(-1)
            ).squeeze(-1)
        
        # Token ratios: s_i(θ) * (π_θ(y_t|x,y<t) / sg[π_θ(y_t|x,y<t)])
        # The second term equals 1 numerically but preserves gradients
        token_ratios = sequence_ratios.unsqueeze(1) * torch.ones_like(current_token_probs)
        
        return sequence_ratios, token_ratios
    
    def gspo_token_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        response_mask: torch.Tensor,
        rewards: torch.Tensor,
        token_advantages: Optional[torch.Tensor],
        group_size: int
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute GSPO-token loss with token-level advantages.
        
        Args:
            token_advantages: Token-level advantages [batch_size, seq_len-1]
                            If None, uses sequence-level advantages
        """
        sequence_ratios, token_ratios = self.compute_token_importance_ratios(
            input_ids, attention_mask, response_mask
        )
        
        if token_advantages is None:
            # Use sequence-level advantages
            advantages = self.compute_group_advantages(rewards, group_size)
            token_advantages = advantages.unsqueeze(1).expand(-1, response_mask.size(1) - 1)
        
        # Apply response mask to token advantages
        shift_response_mask = response_mask[..., 1:].contiguous()
        masked_token_advantages = token_advantages * shift_response_mask
        
        # Apply clipping to token ratios
        clipped_token_ratios = torch.clamp(
            token_ratios,
            1 - self.clip_range,
            1 + self.clip_range
        )
        
        # Compute token-level objectives
        obj1 = token_ratios * masked_token_advantages
        obj2 = clipped_token_ratios * masked_token_advantages
        
        clipped_obj = torch.min(obj1, obj2)
        
        # Average over tokens and responses
        response_lengths = shift_response_mask.sum(dim=-1).clamp(min=1)
        sequence_obj = clipped_obj.sum(dim=-1) / response_lengths
        
        # Average over groups
        batch_size = input_ids.size(0)
        num_groups = batch_size // group_size
        
        grouped_obj = sequence_obj.view(num_groups, group_size)
        group_avg_obj = grouped_obj.mean(dim=1)
        
        loss = -group_avg_obj.mean()
        
        # Statistics
        with torch.no_grad():
            clipping_fraction = (token_ratios != clipped_token_ratios).float().mean()
            mean_token_ratio = token_ratios.mean()
            mean_advantage = masked_token_advantages.sum() / shift_response_mask.sum()
        
        info = {
            'loss': loss.item(),
            'clipping_fraction': clipping_fraction.item(),
            'mean_token_ratio': mean_token_ratio.item(),
            'mean_advantage': mean_advantage.item(),
            'mean_reward': rewards.mean().item(),
        }
        
        return loss, info


# Example usage and utility functions
def create_response_mask(input_ids: torch.Tensor, query_lengths: List[int]) -> torch.Tensor:
    """
    Create mask indicating which tokens are part of the response.
    
    Args:
        input_ids: [batch_size, seq_len]
        query_lengths: List of query lengths for each sample
        
    Returns:
        response_mask: [batch_size, seq_len] with 1s for response tokens
    """
    batch_size, seq_len = input_ids.shape
    response_mask = torch.zeros_like(input_ids, dtype=torch.float)
    
    for i, query_len in enumerate(query_lengths):
        response_mask[i, query_len:] = 1.0
        
    return response_mask


def example_training_loop():
    """
    Example of how to use GSPO trainer.
    """
    # Dummy model setup (replace with actual models)
    vocab_size = 50000
    hidden_size = 768
    
    class DummyLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, hidden_size)
            self.lm_head = nn.Linear(hidden_size, vocab_size)
            
        def forward(self, input_ids, attention_mask=None):
            x = self.embedding(input_ids)
            logits = self.lm_head(x)
            return type('', (), {'logits': logits})()
    
    # Initialize models
    policy_model = DummyLM()
    old_policy_model = DummyLM()
    
    # Initialize trainer
    trainer = GSPOTrainer(
        policy_model=policy_model,
        old_policy_model=old_policy_model,
        clip_range=0.02,
        learning_rate=1e-5
    )
    
    # Dummy training data
    batch_size = 8
    seq_len = 128
    group_size = 4
    
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones_like(input_ids)
    query_lengths = [64] * batch_size  # First 64 tokens are query
    response_mask = create_response_mask(input_ids, query_lengths)
    rewards = torch.rand(batch_size)  # Random rewards for demo
    
    # Training step
    info = trainer.train_step(
        input_ids=input_ids,
        attention_mask=attention_mask,
        response_mask=response_mask,
        rewards=rewards,
        group_size=group_size
    )
    
    print("Training info:", info)
    
    return trainer

if __name__ == "__main__":
    trainer = example_training_loop()