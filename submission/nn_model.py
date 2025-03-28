import torch
import torch.nn as nn
import torch.optim as optim

class NeuralNetworkModel(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=64, output_dim=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(14, 64),  # updated from 10 → 14
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 5)
        )

    def forward(self, x):
        return self.net(x)

    def predict(self, features):
        with torch.no_grad():
            x = torch.tensor(features, dtype=torch.float32)
            return self.forward(x).numpy()

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def save(self, path):
        torch.save(self.state_dict(), path)