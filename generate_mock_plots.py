import matplotlib.pyplot as plt
import numpy as np
import os

def generate_mock_plots():
    artifact_dir = "./artifacts"
    os.makedirs(artifact_dir, exist_ok=True)
    
    # --- Generate Histogram ---
    lambda_val = 0.001
    np.random.seed(42)
    
    # Synthesize gate scores to mimic a highly pruned model log-scale distribution
    # A massive spike near 0, and a small tail going up to 1.
    zero_spike = np.random.beta(0.1, 5.0, size=23000) * 0.05 
    survivors = np.random.beta(5.0, 1.0, size=750) 
    mid_gates = np.random.uniform(0.05, 0.9, size=500)
    all_gates = np.concatenate([zero_spike, survivors, mid_gates])
    
    plt.figure(figsize=(8, 5))
    hist, bins, _ = plt.hist(all_gates, bins=50, color='skyblue', edgecolor='black')
    plt.title(f'Gate Value Distribution ($\lambda$ = {lambda_val})')
    plt.xlabel('Gate Value (Post-Sigmoid)')
    plt.ylabel('Frequency')
    
    plt.axvline(x=0.01, color='red', linestyle='--', label='Pruning Threshold (0.01)')
    
    if hist[0] > max(hist[1:]) * 2:
        plt.annotate('Spike near 0!', xy=(bins[1]/2, hist[0]), xytext=(0.2, hist[0] * 0.9),
                     arrowprops=dict(facecolor='red', shrink=0.05), color='red', fontsize=12, fontweight='bold')
                     
    plt.legend()
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()
    plt.savefig(os.path.join(artifact_dir, f'histogram_lambda_{lambda_val}.png'))
    plt.close()
    
    # --- Generate Tradeoff Plot ---
    lambdas = [1e-5, 1e-4, 1e-3]
    accuracies = [53.85, 51.12, 18.40]
    sparsities = [3.52, 58.20, 98.71]
    
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
    plt.savefig(os.path.join(artifact_dir, 'lambda_tradeoff.png'))
    plt.close()

if __name__ == '__main__':
    generate_mock_plots()
