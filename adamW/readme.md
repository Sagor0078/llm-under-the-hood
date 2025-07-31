## The core innovation of AdamW is decoupling weight decay from the gradient update.

- Original Adam: The original Adam optimizer implements L2 regularization by adding a term to the loss function, which in turn adds a L2_regularization_term * parameters to the gradients. This means the weight decay is integrated into the gradient calculation and scaled by the adaptive learning rate, which the paper "Decoupled Weight Decay Regularization" showed to be suboptimal. This can cause issues with convergence and generalization.

- AdamW (Decoupled Weight Decay): AdamW treats weight decay as a separate step, applied directly to the weights after the main gradient update. The update rule for AdamW looks something like this:

- Gradient Update (like Adam): The parameters are updated using the adaptive moment estimates (first moment and second moment) of the gradients.

- Weight Decay Step: A small portion of the weight is simply decayed away. This is a simple subtraction of learning_rate * weight_decay * weight from the parameter.

This decoupled approach makes the weight decay a form of L2 regularization that behaves correctly with adaptive optimizers, leading to better model generalization and training stability.ea

- [AdamW tutorial](https://www.datacamp.com/tutorial/adamw-optimizer-in-pytorch)