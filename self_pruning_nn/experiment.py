import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from model import SelfPruningNet, BaselineNet, PrunableLinear
from train import get_dataloaders, train_model, evaluate_model, count_parameters, calculate_sparsity_percentage

def plot_histogram(model, lambda_val, save_path):
    all_gates = []
    for module in model.modules():
        if isinstance(module, PrunableLinear):
            gates = torch.sigmoid(module.gate_scores).detach().cpu().numpy().flatten()
            all_gates.extend(gates)
            
    plt.figure(figsize=(8, 5))
    plt.hist(all_gates, bins=50, color='skyblue', edgecolor='black')
    plt.title(f'Gate Value Distribution ($\lambda$ = {lambda_val})')
    plt.xlabel('Gate Value (Post-Sigmoid)')
    plt.ylabel('Frequency')
    
    # Target annotation for the spike near zero
    plt.axvline(x=0.01, color='red', linestyle='--', label='Pruning Threshold (0.01)')
    
    # Annotate the specific spike area if there is strong sparsity
    hist, bins = np.histogram(all_gates, bins=50)
    if hist[0] > max(hist[1:]) * 2: # Spike at zero
        plt.annotate('Spike near 0!', xy=(bins[1]/2, hist[0]), xytext=(0.2, hist[0] * 0.9),
                     arrowprops=dict(facecolor='red', shrink=0.05), color='red', fontsize=12, fontweight='bold')
                     
    plt.legend()
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_tradeoff(lambdas, accuracies, sparsities, save_path):
    fig, ax1 = plt.subplots(figsize=(8, 5))

    color = 'tab:blue'
    ax1.set_xlabel('Lambda ($\lambda$) in log scale')
    ax1.set_ylabel('Test Accuracy (%)', color=color)
    ax1.semilogx(lambdas, accuracies, marker='o', color=color, label='Accuracy', linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, which="both", ls="-", alpha=0.2)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Sparsity (%)', color=color)
    ax2.semilogx(lambdas, sparsities, marker='s', color=color, label='Sparsity', linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.title('Accuracy and Sparsity vs. Sparsity Regularization ($\lambda$)')
    plt.savefig(save_path)
    plt.close()


def run_experiments():
    artifact_dir = "./artifacts"
    os.makedirs(artifact_dir, exist_ok=True)
    
    # Setup device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        
    train_loader, test_loader = get_dataloaders(batch_size=256)
    
    epochs = 10
    results = []
    
    # 1. Train Baseline
    print("--- Running Baseline Model ---")
    baseline_model = BaselineNet()
    baseline_params = count_parameters(baseline_model)
    print(f"Total Parameters (Baseline): {baseline_params}")
    
    trained_baseline = train_model(baseline_model, train_loader, test_loader, epochs=epochs, lambda_val=0.0, device=device, is_baseline=True)
    baseline_acc = evaluate_model(trained_baseline, test_loader, device=device)
    results.append({
        'model': 'Baseline',
        'lambda': 0.0,
        'accuracy': baseline_acc,
        'sparsity': 0.0,
        'params': baseline_params
    })
    
    # 2. Train Prunable Models
    lambda_values = [1e-5, 1e-4, 1e-3]
    accuracies = []
    sparsities = []
    
    for l_val in lambda_values:
        print(f"\n--- Running Prunable Model ($\lambda$ = {l_val}) ---")
        prunable_model = SelfPruningNet()
        
        trained_model = train_model(prunable_model, train_loader, test_loader, epochs=epochs, lambda_val=l_val, device=device, is_baseline=False)
        test_acc = evaluate_model(trained_model, test_loader, device=device)
        sparsity_pct = calculate_sparsity_percentage(trained_model)
        
        accuracies.append(test_acc)
        sparsities.append(sparsity_pct)
        
        results.append({
            'model': 'Self-Pruning',
            'lambda': l_val,
            'accuracy': test_acc,
            'sparsity': sparsity_pct,
            'params': count_parameters(prunable_model)
        })
        
        # Output histogram to the artifact directory for the report
        img_path = os.path.join(artifact_dir, f'histogram_lambda_{l_val}.png')
        plot_histogram(trained_model, l_val, img_path)

    # 3. Save Tradeoff Plot
    tradeoff_path = os.path.join(artifact_dir, 'lambda_tradeoff.png')
    plot_tradeoff(lambda_values, accuracies, sparsities, tradeoff_path)
    
    print("\n--- Final Results ---")
    output_table = f"{'Model':<15} | {'Lambda':<8} | {'Accuracy':<10} | {'Sparsity %':<10} | {'Active Params'}\n"
    output_table += "-" * 65 + "\n"
    
    for res in results:
        active_params = int(res['params'] * (1 - res['sparsity']/100))
        output_table += f"{res['model']:<15} | {res['lambda']:<8} | {res['accuracy']:<10.2f} | {res['sparsity']:<10.2f} | {active_params}\n"
    
    print(output_table)
    
    # Save a quick text log for reference
    with open(os.path.join(artifact_dir, "results_table.txt"), "w") as f:
        f.write(output_table)

if __name__ == '__main__':
    run_experiments()
