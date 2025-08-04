## Speculative Decoding Theory
Speculative decoding is an inference acceleration technique for autoregressive language models that aims to reduce the wall-clock time of text generation without changing the output distribution. The key insight is to use a smaller, faster "draft" model to propose multiple tokens ahead, then use the larger "target" model to verify these proposals in parallel.
Core Concepts:

- Draft Model: A smaller, faster model that generates candidate tokens quickly but with potentially lower quality
- Target Model: The larger, higher-quality model that we actually want to use for generation
- Speculative Execution: The draft model proposes multiple tokens ahead (typically 3-5 tokens)
- Parallel Verification: The target model processes all proposed tokens in a single forward pass
- Acceptance/Rejection: Tokens are accepted based on a probabilistic criterion that maintains the target model's distribution

Mathematical Foundation:
The acceptance probability for a proposed token is:
```text
p_accept = min(1, p_target(x) / p_draft(x))
```
This ensures that the final distribution matches what the target model would produce if run alone.
