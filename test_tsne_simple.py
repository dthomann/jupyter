"""
Simple self-contained TSNE test for low-resource environments
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Generate simple synthetic data
print("Generating synthetic data...")
np.random.seed(42)
n_samples = 200  # Start small
n_features = 50

# Create 3 clusters of data with different centers
n1 = n_samples // 3
n2 = n_samples // 3
n3 = n_samples - n1 - n2

cluster1 = np.random.randn(n1, n_features) + np.array([2] * n_features)
cluster2 = np.random.randn(n2, n_features) + np.array([0] * n_features)
cluster3 = np.random.randn(n3, n_features) + np.array([-2] * n_features)

data = np.vstack([cluster1, cluster2, cluster3])
labels = np.hstack([
    np.zeros(n1),
    np.ones(n2),
    np.ones(n3) * 2
])

print(f"Data shape: {data.shape}")

# First reduce with PCA to make TSNE faster
print("Running PCA...")
pca = PCA(n_components=min(20, n_features), random_state=42)
data_pca = pca.fit_transform(data)
print(f"PCA shape: {data_pca.shape}")

# Try TSNE with memory-efficient settings
print("Running TSNE (this may take a moment)...")
n_subset = data_pca.shape[0]
perplexity_val = min(30, max(5, n_subset // 4))

try:
    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=perplexity_val,
        method='barnes_hut',  # More memory-efficient
        max_iter=300,  # Fewer iterations
        angle=0.5,
        n_jobs=1  # Single-threaded
    )
    data_tsne = tsne.fit_transform(data_pca)
    
    # Plot results
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=labels, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, label='Cluster')
    plt.title(f'TSNE visualization (n={n_samples}, perplexity={perplexity_val})')
    plt.xlabel('TSNE Dimension 1')
    plt.ylabel('TSNE Dimension 2')
    plt.tight_layout()
    plt.savefig('tsne_test.png', dpi=100, bbox_inches='tight')
    print("✓ TSNE completed successfully! Plot saved as 'tsne_test.png'")
    plt.show()
    
except MemoryError:
    print("✗ MemoryError: TSNE failed due to insufficient memory")
    print("Try reducing n_samples or using PCA-only visualization")
    
    # Fallback: just use PCA
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(data_pca[:, 0], data_pca[:, 1], c=labels, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, label='Cluster')
    plt.title('PCA visualization (TSNE fallback)')
    plt.xlabel('PCA Dimension 1')
    plt.ylabel('PCA Dimension 2')
    plt.tight_layout()
    plt.savefig('pca_fallback.png', dpi=100, bbox_inches='tight')
    print("✓ PCA fallback plot saved as 'pca_fallback.png'")
    plt.show()
    
except Exception as e:
    print(f"✗ Error: {e}")
    print(f"Error type: {type(e).__name__}")

print("\nTo test with different sample sizes, modify 'n_samples' at the top of the script")

