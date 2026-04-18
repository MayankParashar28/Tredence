"""
Self-Pruning Neural Network on CIFAR-10
Tredence Analytics AI Engineer Case Study

Author: Mayank
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import numpy as np
import json

# --- 0. Reproducibility ---
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")

# --- 1. PrunableLinear Layer ---

class PrunableLinear(nn.Module):
    """
    Drop-in replacement for nn.Linear that learns to prune its own weights.
    Maintains a learnable gate_score for each weight. During forward pass:
    - pass gate_scores through sigmoid
    - multiply weights by gates
    - apply standard linear operation
    
    The L1 penalty on the gates drives them toward 0, pruning the weights.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias   = nn.Parameter(torch.zeros(out_features))

        # init near 0 so sigmoid ~0.5 (half-open to start)
        self.gate_scores = nn.Parameter(torch.zeros(out_features, in_features))

        # kaiming init since we're using relu later
        nn.init.kaiming_uniform_(self.weight, nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gates = torch.sigmoid(self.gate_scores)
        pruned_weights = self.weight * gates
        return F.linear(x, pruned_weights, self.bias)

    def gate_values(self) -> torch.Tensor:
        """Return the current gate values (detached) for analysis."""
        return torch.sigmoid(self.gate_scores).detach()

    def sparsity(self, threshold: float = 1e-2) -> float:
        """Fraction of weights whose gate is below `threshold`."""
        gates = self.gate_values()
        return (gates < threshold).float().mean().item()


# --- 2. Network Definition ---

class SelfPruningNet(nn.Module):
    """
    MLP for CIFAR-10 (3072 -> 10).
    Using PrunableLinear layers and BatchNorm to keep training stable.
    """

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()

        self.fc1 = PrunableLinear(3072, 1024)
        self.bn1 = nn.BatchNorm1d(1024)

        self.fc2 = PrunableLinear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)

        self.fc3 = PrunableLinear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)

        self.fc4 = PrunableLinear(256, 10)   # 10 CIFAR-10 classes

        self.dropout = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)

        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)

        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)

        x = F.relu(self.bn3(self.fc3(x)))

        x = self.fc4(x)   # raw logits – CrossEntropyLoss handles softmax
        return x

    def prunable_layers(self):
        """Iterate over all PrunableLinear sublayers."""
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                yield module

    def sparsity_loss(self) -> torch.Tensor:
        """
        Calculates L1 norm of all gate values.
        L1 is used instead of L2 to drive gates exactly to 0.
        """
        total = torch.tensor(0.0, device=DEVICE)
        for layer in self.prunable_layers():
            gates = torch.sigmoid(layer.gate_scores)   # keep graph for grad
            total = total + gates.sum()
        return total

    def overall_sparsity(self, threshold: float = 1e-2) -> float:
        """Fraction of pruned weights across the whole network."""
        pruned = total = 0
        for layer in self.prunable_layers():
            gates = layer.gate_values()
            pruned += (gates < threshold).sum().item()
            total  += gates.numel()
        return pruned / total if total > 0 else 0.0

    def all_gate_values(self) -> np.ndarray:
        """Concatenate all gate values into a single 1-D numpy array."""
        vals = []
        for layer in self.prunable_layers():
            vals.append(layer.gate_values().cpu().numpy().ravel())
        return np.concatenate(vals)


# --- 3. Data Loading ---

def get_dataloaders(batch_size: int = 128):
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root="./data", train=True,  download=True, transform=transform_train)
    test_set  = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True,  num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=256,
                              shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, test_loader


# --- 4. Training & Evaluation ---

def train_one_epoch(model, loader, optimizer, lam: float):
    model.train()
    total_loss = cls_loss_sum = sparse_loss_sum = 0.0
    correct = 0

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()

        logits = model(images)

        cls_loss    = F.cross_entropy(logits, labels)
        sparse_loss = model.sparsity_loss()
        loss        = cls_loss + lam * sparse_loss

        loss.backward()
        optimizer.step()

        total_loss      += loss.item()
        cls_loss_sum    += cls_loss.item()
        sparse_loss_sum += sparse_loss.item()
        correct         += (logits.argmax(1) == labels).sum().item()

    n = len(loader.dataset)
    return {
        "loss":         total_loss      / len(loader),
        "cls_loss":     cls_loss_sum    / len(loader),
        "sparse_loss":  sparse_loss_sum / len(loader),
        "train_acc":    correct / n,
    }


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        logits = model(images)
        correct += (logits.argmax(1) == labels).sum().item()
    return correct / len(loader.dataset)


# --- 5. Full Experiment for One Lambda ---

def run_experiment(lam: float, epochs: int = 40, batch_size: int = 128):
    print(f"\n{'='*55}")
    print(f"  λ = {lam}   (sparsity weight)")
    print(f"{'='*55}")

    train_loader, test_loader = get_dataloaders(batch_size)

    model = SelfPruningNet().to(DEVICE)

    # skip weight decay for gate scores to avoid double penalizing
    reg_params  = [p for n, p in model.named_parameters() if "gate_scores" not in n]
    gate_params = [p for n, p in model.named_parameters() if "gate_scores" in n]

    optimizer = optim.Adam([
        {"params": reg_params, "weight_decay": 1e-4},
        {"params": gate_params, "weight_decay": 0.0}
    ], lr=1e-3)

    # cosine annealing scheduler for smoother convergence
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = []

    for epoch in range(1, epochs + 1):
        stats = train_one_epoch(model, train_loader, optimizer, lam)
        scheduler.step()

        if epoch % 5 == 0 or epoch == 1:
            test_acc  = evaluate(model, test_loader)
            sparsity  = model.overall_sparsity()
            print(f"  Epoch {epoch:3d} | "
                  f"Loss {stats['loss']:.4f} | "
                  f"Cls {stats['cls_loss']:.4f} | "
                  f"Sparse {stats['sparse_loss']:.4f} | "
                  f"TrainAcc {stats['train_acc']:.3f} | "
                  f"TestAcc {test_acc:.3f} | "
                  f"Sparsity {sparsity:.1%}")
            history.append({
                "epoch":    epoch,
                "test_acc": test_acc,
                "sparsity": sparsity,
                **stats,
            })

    final_test_acc = evaluate(model, test_loader)
    final_sparsity = model.overall_sparsity()
    gate_vals      = model.all_gate_values()

    print(f"\n  ✔ Final Test Accuracy : {final_test_acc:.2%}")
    print(f"  ✔ Final Sparsity      : {final_sparsity:.2%}")

    return {
        "lam":          lam,
        "test_acc":     final_test_acc,
        "sparsity":     final_sparsity,
        "gate_vals":    gate_vals,
        "history":      history,
        "model":        model,
    }


# --- 6. Plotting ---

def plot_gate_distribution(results: list, save_path: str = "gate_distributions.png"):
    """
    For each λ: histogram of final gate values.
    A well-pruned model shows a large spike near 0 and a secondary cluster
    of "kept" gates distributed away from 0.
    """
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    colors = ["#2196F3", "#4CAF50", "#FF5722"]

    for ax, res, col in zip(axes, results, colors):
        gates = res["gate_vals"]
        ax.hist(gates, bins=80, color=col, alpha=0.85, edgecolor="white", linewidth=0.3)
        ax.axvline(0.01, color="red", linestyle="--", linewidth=1.2, label="Prune threshold (0.01)")
        ax.set_title(
            f"λ = {res['lam']}\n"
            f"Test Acc: {res['test_acc']:.2%}  |  Sparsity: {res['sparsity']:.2%}",
            fontsize=12, fontweight="bold"
        )
        ax.set_xlabel("Gate Value", fontsize=11)
        ax.set_ylabel("Count",      fontsize=11)
        ax.legend(fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Self-Pruning Network – Gate Value Distributions", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Plot saved → {save_path}")


def plot_training_curves(results: list, save_path: str = "training_curves.png"):
    """Test accuracy and sparsity over epochs for each λ."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["#2196F3", "#4CAF50", "#FF5722"]

    for res, col in zip(results, colors):
        epochs   = [h["epoch"]    for h in res["history"]]
        accs     = [h["test_acc"] for h in res["history"]]
        sparsity = [h["sparsity"] for h in res["history"]]

        ax1.plot(epochs, accs,     color=col, marker="o", label=f"λ={res['lam']}")
        ax2.plot(epochs, sparsity, color=col, marker="s", label=f"λ={res['lam']}")

    ax1.set_title("Test Accuracy vs Epoch", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Accuracy")
    ax1.legend(); ax1.grid(alpha=0.3)
    ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)

    ax2.set_title("Sparsity Level vs Epoch", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Sparsity")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax2.legend(); ax2.grid(alpha=0.3)
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved → {save_path}")


# --- 7. Main ---

def main():
    LAMBDAS = [1e-5, 1e-4, 5e-4]   # low / medium / high sparsity pressure
    EPOCHS  = 40

    all_results = []
    for lam in LAMBDAS:
        res = run_experiment(lam=lam, epochs=EPOCHS)
        all_results.append(res)

    # Summary table
    print("\n\n" + "─" * 55)
    print(f"  {'Lambda':<12} {'Test Accuracy':>15} {'Sparsity (%)':>14}")
    print("─" * 55)
    for r in all_results:
        print(f"  {r['lam']:<12} {r['test_acc']:>14.2%} {r['sparsity']:>13.2%}")
    print("─" * 55)

    # Plots
    plot_gate_distribution(all_results, "gate_distributions.png")
    plot_training_curves(all_results,   "training_curves.png")

    # Save results
    summary = [
        {"lambda": r["lam"],
         "test_accuracy": round(r["test_acc"], 4),
         "sparsity_pct":  round(r["sparsity"] * 100, 2)}
        for r in all_results
    ]
    with open("results_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nResults saved → results_summary.json")

    # Best model gate analysis
    best = max(all_results, key=lambda r: r["test_acc"])
    print(f"\nBest model: λ={best['lam']}  "
          f"Test Acc={best['test_acc']:.2%}  "
          f"Sparsity={best['sparsity']:.2%}")


if __name__ == "__main__":
    main()