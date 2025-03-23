import torch.nn as nn
import torch.nn.functional as F

# Define the model architecture
class PokerPolicyNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PokerPolicyNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.output(x)
        return logits  # Raw logits (for masking + softmax later)
