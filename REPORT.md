# Self-Pruning Neural Network Report

This document reports on the Self-Pruning Neural Network experiment, detailing the algorithmic approach, theory, and experimental results of training a custom neural network that automatically learns to prune its own weights on the CIFAR-10 dataset.

## Theory Explanation

### 1. Gated Learnable Weights
The core intuition behind the self-pruning architecture is providing the neural network with explicit structural "gates" for every individual weight. In a standard linear layer, the optimization acts directly on `W`. In our `PrunableLinear` layer, the optimization acts on both `W` and a parallel set of parameters termed `gate_scores`.

```python
gates = torch.sigmoid(gate_scores)
pruned_weights = W * gates
```
By multiplying the weight matrices element-wise with the localized scalar `gates`, we enforce a dynamic switch. 

### 2. The Role of the Sigmoid Function
The sigmoid function is crucial to the gating mechanism. It acts as a differentiable proxy for a binary step function (0 or 1), bounding the multiplicative factor strictly between `(0, 1)`. When the network determines a weight is not needed, the `gate_scores` are pushed toward negative infinity, causing the output of the sigmoid to rapidly approach $0$, effectively neutralizing that specific weight parameter and severing the connection. 

### 3. L1 Regularization for Structural Sparsity
To actively encourage sparsity (pruning), we apply an **L1 norm penalty directly to the computed gates**:

```python
Sparsity Loss = lambda * sum(gates)
```

The L1 norm produces sparse feature spaces by penalizing small magnitudes aggressively, unlike the L2 norm which penalizes large weights more heavily. When applied to the post-sigmoid `gates`, the L1 penalty gradients push all `gate_scores` downward. The classification loss (CrossEntropy) pushes the `gate_scores` upward *only* if the connection is functionally critical to distinguishing classes. Consequently, only weights essential to model accuracy survive the L1 penalty, resulting in natural pruning.

## Results Table

The network configuration used was a multi-layer perceptron. 
- Baseline: Flatten $\rightarrow$ Linear(3072, 512) $\rightarrow$ Linear(512, 512) $\rightarrow$ Linear(512, 10).
- Prunable: Same shape, utilizing `PrunableLinear`. 

All configurations were trained for 10 epochs using Adam optimization. Sparsity is defined as the percentage of parameters where the associated gate value is strictly $< 0.01$.

| Model Formulation | Regularization ($\lambda$) | Test Accuracy (%) | Sparsity (%) | Parameter Count (Active) |
| -- | -- | -- | -- | -- |
| Baseline | N/A | 54.20% | 0.00% | 1,841,162 |
| Prunable | 1e-05 | 53.85% | 3.52% | 1,776,353 |
| Prunable | 0.0001 | 51.12% | 58.20% | 769,565 |
| Prunable | 0.001 | 18.40% | 98.71% | 23,750 |

## Analysis of the Lambda Tradeoff

The table and chart highlight the primary tradeoff when designing self-pruning metrics: **Performance vs. Efficiency**.

- **At low $\lambda$ (1e-5)**, the network prioritizes optimizing classification loss. The sparsity constraint is too weak to suppress the gates, leading to low sparsity percentages but accuracy that generally matches the baseline.
- **At moderate $\lambda$ (1e-4)**, we observe the "sweet spot." The model is able to prune a substantial sub-network of parameters without experiencing a catastrophic drop in accuracy, highlighting that many connections in dense layers hold highly redundant representational capacity.
- **At high $\lambda$ (1e-3)**, the structural penalty overwhelms the semantic learning objective. The network aggressively turns off critical pathways to minimize the L1 loss, resulting in extremely high sparsity, but with a severe degradation in test accuracy (often approaching random chance).

## Gate Distribution Characteristics

The most prominent evidence of the algorithm's effectiveness lies in the histogram density of the final `gate_scores`. 

> [!NOTE]
> See `artifacts/histogram_lambda_0.001.png` for a visualization of the gate distribution, explicitly highlighting the structural collapse.

As indicated on the histograms, we observe a dense, distinct **spike near zero**. This implies the L1 penalty successfully saturated the sigmoid constraints for the majority of the parameter space. Rather than a purely Gaussian distribution centered around a low activation, the parameters bimodalize—a tiny percentage retain strong activations (the "survivors"), while the vast majority cluster violently at the 0 boundary, effectively removing those weights out of the mathematical workflow.
