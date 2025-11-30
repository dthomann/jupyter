# Simple TSNE test cell - paste this into a Jupyter notebook cell
# Notebook-safe version with explicit memory management

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import gc

# Clear any existing figures to free memory
plt.close('all')

# Generate simple synthetic data
print("Generating synthetic data...")
np.random.seed(42)
n_samples = 150  # Reduced for notebook safety
n_features = 50

# Create 3 clusters with different centers
n1 = n_samples // 3
n2 = n_samples // 3
n3 = n_samples - n1 - n2

cluster1 = np.random.randn(n1, n_features) + np.array([2] * n_features)
cluster2 = np.random.randn(n2, n_features) + np.array([0] * n_features)
cluster3 = np.random.randn(n3, n_features) + np.array([-2] * n_features)

data = np.vstack([cluster1, cluster2, cluster3])
labels = np.hstack([np.zeros(n1), np.ones(n2), np.ones(n3) * 2])

# Clean up intermediate variables
del cluster1, cluster2, cluster3
gc.collect()

print(f"Data shape: {data.shape}")

# PCA first
print("Running PCA...")
pca = PCA(n_components=min(20, n_features), random_state=42)
data_pca = pca.fit_transform(data)

# Clean up original data after PCA
del data
gc.collect()

# TSNE with memory-efficient settings
print("Running TSNE...")
n_subset = data_pca.shape[0]
perplexity_val = min(30, max(5, n_subset // 4))
print(f"  Samples: {n_subset}, Perplexity: {perplexity_val}")

try:
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity_val,
                method='barnes_hut', max_iter=250, angle=0.5, n_jobs=1,
                verbose=1)  # verbose=1 shows progress
    data_tsne = tsne.fit_transform(data_pca)

    # Clean up PCA data after TSNE
    del data_pca
    gc.collect()

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        data_tsne[:, 0], data_tsne[:, 1], c=labels, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, label='Cluster', ax=ax)
    ax.set_title(f'TSNE Test (n={n_samples}, perplexity={perplexity_val})')
    ax.set_xlabel('TSNE Dim 1')
    ax.set_ylabel('TSNE Dim 2')
    plt.tight_layout()
    plt.show()

    # Clean up
    del data_tsne
    plt.close(fig)
    gc.collect()

    print("✓ Success! If this worked, try increasing n_samples gradually (200, 300, 500...)")

except MemoryError as e:
    print(f"✗ MemoryError: {e}")
    print("Try reducing n_samples or restarting the kernel")
except Exception as e:
    print(f"✗ Error: {e}")
    print(f"Error type: {type(e).__name__}")
