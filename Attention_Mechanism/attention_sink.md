## Attention Sink 
In modern large language models (LLMs), an attention sink is a phenomenon where the model disproportionately allocates attention to the first few tokens of a sequence, even if those tokens have no semantic importance. The theory behind attention sinks suggests they serve a crucial, functional role in the model's stability and performance, particularly in long-sequence and streaming applications.

## How It Improves Performance

The most significant benefit of attention sinks is their role in enabling efficient streaming LLMs. Traditional LLMs use a "KV cache" to store the key and value states of all previous tokens. This cache grows linearly with the sequence length, consuming an immense amount of memory and making it unfeasible for long dialogues or documents.

- Window Attention Failure: A common approach to mitigate this is window attention, where only the most recent tokens are kept in the cache. However, models using this method often experience a sharp drop in performance and fluency as soon as the first few tokens (the implicit attention sinks) are evicted from the window.

- The StreamingLLM Solution: The theory of attention sinks led to the development of the StreamingLLM framework. This framework modifies the window attention approach by permanently retaining the first few tokens—the attention sinks—in the cache, alongside a sliding window of the most recent tokens.  This design allows the model to:

- Maintain a constant memory footprint, regardless of the overall sequence length.

- Preserve the crucial "offloading" mechanism provided by the attention sinks.

- Generalize to effectively "infinite" context lengths without fine-tuning, as it avoids the performance collapse that happens when the initial tokens are discarded.

In essence, attention sinks are a learned behavior that provides a stabilization mechanism for autoregressive LLMs, and by explicitly keeping them in the cache, engineers can build more memory-efficient and stable models for real-world streaming applications like multi-turn chats.

Here's a video that explains how this phenomenon makes LLMs usable for practical applications.

- [Attention Sink: The Fluke That Made LLMs Actually Usable](https://www.youtube.com/watch?v=Y8Tj9kq4iWY)
- [Attention Sink](https://github.com/sail-sg/Attention-Sink)