## Parameter-Efficient Fine-Tuning (PEFT) 

Parameter-Efficient Fine-Tuning (PEFT) is a collection of techniques designed to adapt large pre-trained models (especially Large Language Models or LLMs) to new, specific tasks or datasets by updating only a small subset of their parameters, rather than fine-tuning the entire model. The goal is to achieve performance comparable to full fine-tuning, but with significantly reduced computational resources, memory footprint, and storage requirements.

### Motivation: Why PEFT is Crucial for LLMs

The rise of colossal LLMs (with billions or even trillions of parameters) has brought unprecedented capabilities but also presents significant challenges for fine-tuning and deployment:

- Prohibitive Computational Cost: Full fine-tuning involves updating every single parameter of the model. For models like GPT-3 (175B parameters), this requires immense GPU memory (hundreds of GBs) and computational power, making it inaccessible to most researchers and organizations.
- Memory and Storage Overhead: Each task-specific fine-tuned version of a large LLM would require storing a full copy of the model (hundreds of GBs per copy). This quickly becomes unmanageable for deploying multiple specialized models.
- Time Consumption: The extensive computations involved in full fine-tuning translate to very long training times, hindering rapid experimentation and iteration.
- Catastrophic Forgetting: While fine-tuning specializes a model, it can sometimes lead to "catastrophic forgetting," where the model loses some of its general knowledge acquired during pre-training. PEFT methods often mitigate this by preserving the majority of the pre-trained weights.
- Accessibility: PEFT democratizes access to state-of-the-art LLMs by making fine-tuning feasible on more modest hardware, including consumer-grade GPUs.

### General Principles of PEFT Methods:

Most PEFT techniques operate on one or more of the following principles:

- Freezing Most Parameters: The vast majority of the pre-trained model's parameters are kept frozen (their weights are not updated during fine-tuning). This preserves the foundational knowledge of the base model and drastically reduces the number of parameters for which gradients need to be computed and optimizer states maintained.
- Introducing Few Trainable Parameters: Instead of modifying existing parameters, PEFT methods often introduce a small number of new, trainable parameters (e.g., in the form of small adapter modules or low-rank matrices) into the model's architecture. These new parameters are specifically designed to capture task-specific knowledge.
- Reparameterization: Some methods reparameterize the weight updates in a lower-dimensional space, effectively allowing for efficient updates to large matrices through a smaller set of trainable parameters.
- Soft Prompts/Prefixes: Instead of modifying the model's internal weights, some techniques optimize continuous vectors that are prepended or injected into the input sequence or internal activations, guiding the model's behavior for a specific task.

### Common Parameter-Efficient Fine-Tuning (PEFT) Techniques:

Here are some of the most prominent PEFT techniques:

LoRA (Low-Rank Adaptation):

- Concept: LoRA is based on the "low-rank hypothesis," which states that the changes needed to adapt a large pre-trained model to a new task can be effectively represented by low-rank matrices.
- Mechanism: For selected large weight matrices (e.g., in attention layers), LoRA freezes the original pre-trained weights (W0​) and injects two small, trainable matrices, A and B, such that their product (BA) approximates the desired weight update (ΔW). The forward pass becomes h=(W0​+BA)x. Only A and B are trained.
- Benefits: Drastically reduces trainable parameters, memory usage, and storage. Can be merged into the base model for zero inference latency.

QLoRA (Quantized Low-Rank Adaptation):

- Concept: An extension of LoRA that combines its parameter efficiency with 4-bit quantization of the base pre-trained model.
- Mechanism: The entire base model is quantized to 4-bit precision (e.g., using NF4 quantization) and kept frozen. LoRA adapters (matrices A and B) are added in higher precision (e.g., FP16/BF16) and are the only trainable parameters. During computation, the 4-bit weights are de-quantized on-the-fly to a higher precision. QLoRA also introduces "Double Quantization" and "Paged Optimizers" for further memory savings.
- Benefits: Enables fine-tuning of massive LLMs (e.g., 65B parameters) on a single consumer-grade GPU, making fine-tuning highly accessible and cost-effective.

Adapter Tuning:

- Concept: Inserts small, trainable "adapter modules" between layers of the frozen pre-trained model.
- Mechanism: Each adapter is typically a bottleneck architecture (down-projection, non-linearity, up-projection). Only the parameters within these small adapter modules are trained.
- Benefits: Modular, allows for easy task-switching by loading different adapters. Can be inserted in series or parallel. May introduce slight inference latency if not merged.

Prompt Tuning / Prefix-Tuning:

- Concept: Instead of modifying the model's weights, these methods optimize continuous "soft prompt" vectors that are prepended or added to the input embeddings (Prompt Tuning) or to the key and value matrices in each Transformer layer (Prefix-Tuning).
- Mechanism: The base model remains entirely frozen. Only the parameters of these "soft prompt" vectors are trained. The model learns to condition its behavior on these learned prompts.
- Benefits: Extremely parameter-efficient (often only a few hundred or thousand trainable parameters). No model weight updates required. Can be very effective for generative tasks.

BitFit:

- Concept: A very simple selective fine-tuning method.
- Mechanism: Only the bias terms in the pre-trained model are updated, while all other weights are frozen.
- Benefits: Extremely few trainable parameters, very memory-efficient. Surprisingly effective for some tasks.


PEFT has become an indispensable set of tools for working with large language models, making their customization and deployment practical and efficient.