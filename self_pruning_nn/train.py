import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import PrunableLinear

def get_dataloaders(batch_size=128):
    """
    Load CIFAR-10 data loaders.
    """
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, test_loader

def compute_sparsity_loss(model):
    """
    Calculate the L1 norm of all gate values (post-sigmoid).
    This encourages the gates to approach 0.
    """
    sparsity_loss = 0.0
    for module in model.modules():
        if isinstance(module, PrunableLinear):
            gates = torch.sigmoid(module.gate_scores)
            sparsity_loss += torch.sum(gates)
    return sparsity_loss

def calculate_sparsity_percentage(model, threshold=1e-2):
    """
    A weight is considered pruned if its gate is < 1e-2. 
    Returns the percentage of pruned weights out of total prunable weights.
    """
    pruned_weights = 0
    total_weights = 0
    for module in model.modules():
        if isinstance(module, PrunableLinear):
            gates = torch.sigmoid(module.gate_scores).detach().cpu()
            pruned_weights += torch.sum(gates < threshold).item()
            total_weights += gates.numel()
    
    if total_weights == 0:
        return 0.0
    return (pruned_weights / total_weights) * 100.0

def count_parameters(model):
    """Counts the total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_model(model, train_loader, test_loader, epochs=10, lambda_val=0.0, device='cpu', is_baseline=False):
    """
    Full training loop incorporating the custom lambda-weighted sparsity loss along with Standard CrossEntropy. 
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_acc = 0.0
    
    print(f"Training on: {device} | Lambda: {lambda_val} | Epochs: {epochs} | Baseline: {is_baseline}")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            class_loss = criterion(outputs, targets)
            
            if is_baseline:
                loss = class_loss
            else:
                sp_loss = compute_sparsity_loss(model)
                loss = class_loss + lambda_val * sp_loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        # Eval after epoch
        test_acc = evaluate_model(model, test_loader, device)
        scheduler.step()
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), f'best_model_lambda_{lambda_val}.pth')
            
        if not is_baseline:
            sparsity_pct = calculate_sparsity_percentage(model)
            print(f"Epoch {epoch+1:02d}/{epochs} - Total Loss: {total_loss/len(train_loader):.4f} - Test Acc: {test_acc:.2f}% - Sparsity: {sparsity_pct:.2f}%")
        else:
            print(f"Epoch {epoch+1:02d}/{epochs} - Total Loss: {total_loss/len(train_loader):.4f} - Test Acc: {test_acc:.2f}%")

    return model

def evaluate_model(model, test_loader, device='cpu'):
    """Evaluates generic model accuracy."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
    return 100.0 * correct / total
