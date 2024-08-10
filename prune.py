import torch
import torch.nn as nn
import torch.nn.utils.prune as prune


# Define a simple feedforward neural network
class PrunableNN(nn.Module):
    def __init__(self):
        super(PrunableNN, self).__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def apply_pruning(model, pruning_percent):
    # Apply weight pruning to linear layers
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=pruning_percent)
            if module.bias is not None:
                prune.l1_unstructured(module, name='bias', amount=pruning_percent)


def remove_pruning(model):
    # Remove pruning reparametrization to finalize the model
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            prune.remove(module, 'weight')
            if module.bias is not None:
                prune.remove(module, 'bias')


def print_model_weights(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            print(f"\n{name} weights:")
            print(module.weight.data)
            if module.bias is not None:
                print(f"{name} bias:")
                print(module.bias.data)


# Initialize and print original model
model = PrunableNN()
print("Original Model:")
print(model)
print_model_weights(model)

# Apply pruning and print model state
apply_pruning(model, pruning_percent=0.3)
print("\nModel after Pruning:")
print(model)
print_model_weights(model)

# Optionally, remove pruning reparametrization and print final model state
remove_pruning(model)
print("\nModel after Removing Pruning Reparametrization:")
print(model)
print_model_weights(model)
