import numpy as np
import matplotlib
matplotlib.use('Agg')  # I'm using WSL
import matplotlib.pyplot as plt
from scipy.stats import norm

def plot_gmm(mu, var, pi, y, example_index, name):
    plt.figure(figsize=(10, 6))
    
    # Convert tensors to numpy arrays
    mu = mu[example_index].detach().cpu().numpy()
    var = var[example_index].detach().cpu().numpy()
    pi = pi[example_index].detach().cpu().numpy()
    y_true = y[example_index]

    # Create x values for plotting
    x = np.linspace(mu.min() - 3*np.sqrt(var.max()), 
                    mu.max() + 3*np.sqrt(var.max()), 1000)

    # Plot each Gaussian component
    gmm = np.zeros_like(x)
    for i in range(len(mu)):
        component = pi[i] * norm.pdf(x, mu[i], np.sqrt(var[i]))
        plt.plot(x, component, label=f'Component {i+1}')
        gmm += component

    # Plot the sum of all components
    plt.plot(x, gmm, 'k-', linewidth=2, label='Mixture')

    # Plot the true value
    plt.axvline(y_true, color='r', linestyle='--', label='True Value')

    plt.title(f'Gaussian Mixture Model - Example {example_index}')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.savefig(name)