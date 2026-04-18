import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PrunableLinear(nn.Module):
    """
    Custom linear layer that learns to prune its own weights during training.
    Each weight has an associated learnable gate score.
    """
    def __init__(self, in_features, out_features):
        super(PrunableLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Standard parameters
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        
        # Learnable gating parameter (same shape as weight)
        self.gate_scores = nn.Parameter(torch.Tensor(out_features, in_features))
        
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize `weight` and `bias` identically to `nn.Linear`
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in = self.weight.size(1)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)
        
        # Initialize `gate_scores` strongly positive so doors start fully "open" 
        # (sigmoid(1.0) ~ 0.73, sigmoid(3.0) ~ 0.95). L1 loss will force them down if unneeded.
        nn.init.normal_(self.gate_scores, mean=1.0, std=0.1)

    def forward(self, x):
        # Apply sigmoid to limit gates perfectly in (0, 1) bound
        gates = torch.sigmoid(self.gate_scores)
        
        # Element-wise multiply gates with weights (differentiable for both)
        pruned_weights = self.weight * gates
        
        # Calculate standard affine transform using the "pruned" weights
        return F.linear(x, pruned_weights, self.bias)

class SelfPruningNet(nn.Module):
    """
    MLP that uses the custom PrunableLinear layers inside.
    Flatten mapping: (B, 3, 32, 32) -> (B, 3072)
    """
    def __init__(self, hidden_dim=512):
        super(SelfPruningNet, self).__init__()
        self.flatten = nn.Flatten()
        # 3 PrunableLinear stages
        self.fc1 = PrunableLinear(3072, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = PrunableLinear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.fc3 = PrunableLinear(hidden_dim, 10)
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x

class BaselineNet(nn.Module):
    """
    A standard MLP configured exactly like SelfPruningNet but utilizing primitive nn.Linear layers.
    Used exclusively as a comparison unpruned baseline.
    """
    def __init__(self, hidden_dim=512):
        super(BaselineNet, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(3072, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, 10)
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x
