# LLM Under The Hood - Overview

| Module/Folder         | Files Included                                                                 | Description                                                  |
|-----------------------|----------------------------------------------------------------------------------|--------------------------------------------------------------|
| [Attention_Mechanism](https://github.com/Sagor0078/llm-under-the-hood/tree/main/Attention_Mechanism) | `main.py`, `PagedAttention.py`, `Sliding_window.py`, `readme.md` | Core attention mechanisms including sliding window and paged attention |
| [KV_Cache](https://github.com/Sagor0078/llm-under-the-hood/tree/main/KV_Cache)                         | `main.py`, `readme.md`                                      | Key-Value caching strategies in Transformers                |
| [adamW](https://github.com/Sagor0078/llm-under-the-hood/tree/main/adamW)                               | `adam.py`, `readme.md`                                      | AdamW optimizer implementation and theory                   |
| [flashAttention](https://github.com/Sagor0078/llm-under-the-hood/tree/main/flashAttention)             | `flush_attention_triton.py`, `readme.md`                    | FlashAttention kernel using Triton                         |
| [MoE](https://github.com/Sagor0078/llm-under-the-hood/tree/main/MoE)                                   | `readme.md`                                                 | Mixture of Experts (MoE) techniques in LLMs                |
| [optimizing_llm_inference](https://github.com/Sagor0078/llm-under-the-hood/tree/main/optimizing_llm_inference) | `batching.py`, `speculative_decoding.py`, `readme.md`        | Techniques for efficient inference (e.g., batching, speculative decoding) |
| [PEFT](https://github.com/Sagor0078/llm-under-the-hood/tree/main/PEFT)                                 | `LoRA.py`, `readme.md`                                       | Parameter-Efficient Fine-Tuning (LoRA)                      |
| [PromptTechnique](https://github.com/Sagor0078/llm-under-the-hood/tree/main/PromptTechnique)           | `main.py`, `readme.md`                                      | Prompting strategies and techniques                        |
| [Quantization](https://github.com/Sagor0078/llm-under-the-hood/tree/main/Quantization)                 | `main.py`, `readme.md`                                      | Model quantization for memory-efficient inference           |
| [RLHF](https://github.com/Sagor0078/llm-under-the-hood/tree/main/RLHF)                                 | `GRPO.py`, `GSPO.py`, `PPO.py`, `readme.md`                 | Reinforcement Learning from Human Feedback (GRPO, GSPO, PPO) |
| [RMSNorm](https://github.com/Sagor0078/llm-under-the-hood/tree/main/RMSNorm)                           | `main.py`, `readme.md`, `RMSNorm.png`, `rmenorm_formula.png`| Root Mean Square Layer Normalization                        |
| [RoPE](https://github.com/Sagor0078/llm-under-the-hood/tree/main/RoPE)                                 | `main.py`, `readme.md`, `rope.png`                          | Rotary Positional Embeddings                               |
| [SwiGLU](https://github.com/Sagor0078/llm-under-the-hood/tree/main/SwiGLU)                             | `main.py`, `readme.md`                                      | SwiGLU activation function                                 |
| [Tokenizer](https://github.com/Sagor0078/llm-under-the-hood/tree/main/Tokenizer)                       | `main.py`, `readme.md`                                      | Tokenization logic and concepts                            |
| [LICENSE](https://github.com/Sagor0078/llm-under-the-hood/blob/main/LICENSE)                           | `LICENSE`                                                    | Open source license (MIT)                                  |
| [README.md](https://github.com/Sagor0078/llm-under-the-hood/blob/main/README.md)                       | `README.md`                                                  | Main project overview and navigation                       |
               

> I'm still learning more advanced topics like inference strategies (Distillation, Compilation, Sharding, etc.) coming soon!
> [My profile on LeetGPU](https://leetgpu.com/profile?display_name=Sagor0078)

## External Learning Resources

| Resource | Link | Description |
|---------|------|-------------|
| Stanford CS336 – LLMs From Scratch | [🔗 CS336 Course Page](https://web.stanford.edu/class/cs336/) | Advanced graduate course on large language models from scratch |
| Umar Jamil’s LLM Tutorial Series | [YouTube Playlist](https://www.youtube.com/@umarjamilai/featured) | Step-by-step LLM implementation tutorials |
| Andrej Karpathy – GPT From Scratch | [YouTube Video](https://www.youtube.com/watch?v=kCc8FmEb1nY) | Karpathy's famous transformer explainer video |

