## Reinforcement Learning from Human Feedback (RLHF), Proximal Policy Optimization (PPO), and Group Relative Policy Optimization (GRPO)

These techniques are at the forefront of aligning Large Language Models (LLMs) with human preferences and improving their performance on complex tasks. They build upon the principles of Reinforcement Learning (RL) to refine model behavior.

---
## GRPO vs GSPO: Comprehensive Comparison

### Fundamental Philosophy
GRPO (Group Relative Policy Optimization)

- Token-level approach: Applies importance sampling at individual token level
- Misaligned units: Token-level optimization with sequence-level rewards
- High variance: Each token gets different importance weights, leading to instability

GSPO (Group Sequence Policy Optimization)

- Sequence-level approach: Applies importance sampling at full sequence level
- Aligned units: Sequence-level optimization matching sequence-level rewards
- Lower variance: All tokens in a sequence get equal treatment

### Importance Ratio Definition
GRPO

```math 

Token-level importance ratio:
w_{i,t}(θ) = π_θ(y_{i,t}|x, y_{i,<t}) / π_{θ_old}(y_{i,t}|x, y_{i,<t})

Applied to each token individually
```
GSPO
```math
J_GRPO(θ) = E[1/G ∑_{i=1}^G 1/|y_i| ∑_{t=1}^{|y_i|} min(w_{i,t}(θ)Â_{i,t}, clip(w_{i,t}(θ))Â_{i,t})]

Where: Â_{i,t} = Â_i (same advantage for all tokens in sequence)
```
### Mathematical Objectives
GRPO Objective
```math 
J_GRPO(θ) = E[1/G ∑_{i=1}^G 1/|y_i| ∑_{t=1}^{|y_i|} min(w_{i,t}(θ)Â_{i,t}, clip(w_{i,t}(θ))Â_{i,t})]

Where: Â_{i,t} = Â_i (same advantage for all tokens in sequence)
```

GSPO Objective
```math

J_GSPO(θ) = E[1/G ∑_{i=1}^G min(s_i(θ)Â_i, clip(s_i(θ))Â_i)]

Single importance ratio per sequence
```
### Gradient Analysis

- GRPO Gradient : 
```math
∇_θ J_GRPO = E[1/G ∑_i Â_i · 1/|y_i| ∑_t (π_θ(y_{i,t}|x,y_{i,<t})/π_{θ_old}(y_{i,t}|x,y_{i,<t})) ∇_θ log π_θ(y_{i,t}|x,y_{i,<t})]
```
Problem: Each token gets different weight based on its individual importance ratio

- GSPO Gradient : 
```math
∇_θ J_GSPO = E[1/G ∑_i (π_θ(y_i|x)/π_{θ_old}(y_i|x))^{1/|y_i|} Â_i · 1/|y_i| ∑_t ∇_θ log π_θ(y_{i,t}|x,y_{i,<t})]
```
Advantage: All tokens in sequence get equal weight

### Clipping Mechanism
GRPO

Unit: Individual tokens are clipped
Range: Typically [1-ε, 1+ε] where ε ≈ 0.2
Problem: Accumulates variance across tokens in long sequences

GSPO

Unit: Entire sequences are clipped
Range: Typically much smaller, e.g., [1-ε, 1+ε] where ε ≈ 0.02
Benefit: Sequence-level decision, more stable

### Training Stability
GRPO Issues

Model Collapse: Frequent catastrophic failures, especially with large models
MoE Problems: Expert routing changes cause token-level importance ratios to fluctuate wildly
Long Sequence Issues: Variance accumulates with sequence length
Irreversible Failures: Once collapsed, often can't recover even from checkpoints

GSPO Advantages

Stable Training: No reported model collapse issues
MoE Compatible: Works naturally with MoE models without special techniques
Length Robust: Length normalization handles variable sequence lengths
Recoverable: More graceful handling of training instabilities

### MoE (Mixture of Experts) Compatibility
GRPO + MoE : 
```text
# Requires "Routing Replay" strategy
# Cache activated experts from old policy
# Force new policy to use same expert routing
# Additional memory and computation overhead
```
Problems:

Expert activation changes between policy updates
~10% of experts change for same input after gradient update
Token-level importance ratios become invalid
Requires complex workarounds
GSPO + MoE : 
```text 
# Works naturally without modifications
# No routing replay needed
# Full model capacity utilization
```
Advantages:

Sequence-level ratios are robust to expert routing changes
No artificial constraints on model capacity
Simplified training infrastructure

### Implementation Complexity
GRPO : 
```python
# For each token in each sequence:
for i in range(G):  # Group size
    for t in range(len(y_i)):  # Sequence length
        w_i_t = pi_theta(y_i_t) / pi_old(y_i_t)  # Token importance
        clipped_w = clip(w_i_t, 1-eps, 1+eps)
        loss += min(w_i_t * A_i, clipped_w * A_i)

# Additional complexity for MoE:
# - Routing replay mechanism
# - Expert caching
# - Memory management
```

GSPO:
```python

# For each sequence:
for i in range(G):  # Group size
    s_i = (pi_theta(y_i) / pi_old(y_i)) ** (1/len(y_i))  # Sequence importance
    clipped_s = clip(s_i, 1-eps, 1+eps)
    loss += min(s_i * A_i, clipped_s * A_i)

# No special handling needed for MoE
```

### Clipping Statistics
GRPO

Clipping Fraction: ~0.1% of tokens typically clipped
Few Clipped Samples: Most training data used

GSPO

Clipping Fraction: ~15% of tokens clipped (2 orders of magnitude more!)
Counter-intuitive Result: Despite clipping more, achieves better training efficiency
Interpretation: GRPO's unclipped tokens contain noisy gradients; GSPO's selective clipping removes bad sequences entirely

### Performance Comparison
Training Efficiency

GSPO: Superior training efficiency and faster convergence
GRPO: Slower convergence, requires careful hyperparameter tuning

Benchmark Performance
Based on paper results (Qwen3-30B-A3B-Base):

AIME'24: GSPO achieves higher Pass@1 scores
LiveCodeBench: Better performance across evaluation periods
CodeForces: Higher Elo ratings

Infrastructure Benefits

GSPO: Can potentially use inference engine likelihoods directly
GRPO: Requires recomputation with training engine for precision

### Theoretical Justification
GRPO Problems

Importance Sampling Violation: Token-level weights don't follow importance sampling principles
Single Sample Issue: Each token position has only one sample, can't correct distribution mismatch
Accumulating Variance: Noise compounds across sequence length

GSPO Solutions

Proper Importance Sampling: Sequence-level ratios follow theoretical principles
Multiple Sequences: Group of sequences allows proper distribution correction
Variance Reduction: Length normalization and sequence-level approach reduce noise

### Practical Recommendations
When to Use GRPO

Legacy Systems: If already implemented and working
Short Sequences: Less problematic with shorter responses
Dense Models: Fewer issues without MoE complexity

When to Use GSPO

New Implementations: Recommended for new RL training systems
Large Scale Training: Better stability for large models
MoE Models: Essential for Mixture-of-Experts architectures
Long Sequences: Superior handling of lengthy responses
Production Systems: More reliable for critical applications

Summary
GSPO addresses fundamental theoretical and practical issues in GRPO by:

Aligning optimization unit with reward unit (sequence-level)
Following proper importance sampling principles
Providing natural MoE compatibility
Achieving superior stability and efficiency

The key insight is that the unit of optimization should match the unit of reward - since rewards are given to complete sequences, optimization should also operate at the sequence level rather than individual tokens.

---

## Reference
- [Reinforcement Learning from Human Feedback explained with math derivations and the PyTorch code by Omar Jamil](https://www.youtube.com/watch?v=qGyFrqc34yc&t=300s)
- [Group Relative Policy Optimization (GRPO)](https://www.youtube.com/watch?v=Yi1UCrAsf4o)
- [DeepSeek R1 Theory Overview | GRPO + RL + SFT](https://www.youtube.com/watch?v=QdEuh2UVbu0)
- [Reinforcement Learning from Human Feedback (RLHF), Proximal Policy Optimization (PPO), and Group Relative Policy Optimization (GRPO)](https://g.co/gemini/share/789b35d3de43)
- [Proximal Policy Optimization Algorithms" by John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov (2017)](https://arxiv.org/abs/1707.06347)
- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models" by Shao et al. (2024)](https://arxiv.org/abs/2402.03300)
- [Group Sequence Policy Optimization](https://arxiv.org/abs/2507.18071?fbclid=IwY2xjawL0Xp5leHRuA2FlbQIxMABicmlkETF3OW1mWFBkOTNGcW5ZMUVLAR4_R1-6dMTdqGgE7d3-hcN-CGryAGSu8xNUjEzSGHMI9Y7LhHEKPBgwURPLBQ_aem_zYZLzJmsv-VXFETGKMaQfw)

---