import torch
from torch.optim import AdamW

# Assume you have a model and a dataset
model = YourModel() 
optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)

# Training loop
for epoch in range(num_epochs):
    for batch in data_loader:
        # Forward pass
        inputs, labels = batch
        outputs = model(inputs)
        loss = your_loss_function(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Optimizer step
        optimizer.step()

# lr (learning rate): The step size for the optimizer. A common starting point is 1e-4 or 5e-5 for fine-tuning.

# weight_decay: The coefficient for the decoupled weight decay. A typical value is 0.01.