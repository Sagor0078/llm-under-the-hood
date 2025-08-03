## Continuous Batching

Continuous batching, also known as dynamic or in-flight batching, is a technique that significantly improves the throughput and efficiency of LLM inference by processing requests at the token level, not the request level. Unlike static batching, which waits for a fixed number of requests to be ready before processing them together, continuous batching dynamically adds and removes requests from a batch as they arrive and complete.

### Theory of Continuous Batching

The core problem continuous batching solves is the inefficiency of traditional batching methods for LLM inference. LLM inference consists of two phases:

- Prefill phase: The input prompt is processed in parallel to generate the initial key-value (KV) cache.

- Decoding phase: The model generates one new token at a time, autoregressively, with each new token being added to the input for the next step.

In static batching, all requests in a batch must wait for the longest request to finish generating all its tokens, leading to wasted GPU cycles. Requests with shorter prompts or faster generations are stalled, and the GPU sits idle waiting for the entire batch to complete.

Continuous batching overcomes this by:

- **Iteration-level scheduling:** The batch is not fixed. At each decoding step, the scheduler checks for new incoming requests and for requests that have finished. It removes completed requests from the batch and adds new ones to keep the GPU busy.

- **Dynamic batch sizing:** The batch size changes with each iteration, based on the current workload and available GPU memory.

- **KV Cache Management:** A sophisticated memory management system, often using a technique like **PagedAttention**, is crucial. This system efficiently allocates and frees memory for the KV cache of each sequence, preventing fragmentation and allowing for non-contiguous memory allocation.

This approach keeps the GPU consistently occupied, maximizing throughput and reducing the latency for shorter requests, as they no longer have to wait for longer ones to finish.

Of course. Here are some of the best resources for learning about continuous batching, from theoretical overviews to practical implementation and libraries.

### Core Concepts & Theory

- vLLM Paper & Blog Post: The vLLM project introduced the concept of PagedAttention, which is a key technical component enabling continuous batching. Reading their paper or the original blog post will give you a deep understanding of the memory management and scheduling challenges that continuous batching solves. It explains why traditional methods are inefficient and how PagedAttention offers a more elegant solution.

- BentoML's LLM Inference Handbook: This resource provides a clear and concise explanation of the differences between static, dynamic, and continuous batching. It uses helpful analogies (like a printer or a bus) to make the concepts easy to grasp, and it highlights why continuous batching is the state-of-the-art for LLM inference.

- Databricks Blog Post: This article on LLM inference performance engineering offers a great high-level overview of the metrics that matter (TTFT, TPOT, Latency, Throughput) and how batching techniques, including continuous batching, affect them. It provides a solid foundation for understanding the "why" behind these optimizations.

Practical Implementation & Frameworks

- vLLM Library: This is the gold standard for continuous batching. If you're serious about running fast, high-throughput LLM inference, vLLM is the most powerful and well-regarded open-source option. Their documentation is an excellent resource for learning how to use it, deploy it, and understand its core features. It is built on top of PyTorch and provides highly optimized CUDA kernels to make continuous batching work efficiently.[vLLM github](https://github.com/vllm-project/vllm)

### References
- [Scaling LLMs with Batch Processing: Ultimate Guide](https://latitude-blog.ghost.io/blog/scaling-llms-with-batch-processing-ultimate-guide/)
- [Static, dynamic and continuous batching | LLM Inference Handbook](https://bentoml.com/llm/inference-optimization/static-dynamic-continuous-batching)
- [LLM Inference Performance Engineering: Best Practices](https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices)

