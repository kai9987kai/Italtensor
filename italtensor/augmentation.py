"""SMOTE (Synthetic Minority Over-sampling Technique) for balancing class distributions."""

from __future__ import annotations

import numpy as np


def apply_smote(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    k_neighbors: int = 3,
    target_ratio: float = 1.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Oversample the minority class in a binary classification dataset using SMOTE.
    
    Args:
        features: 2D array of shape (N, D) containing sample features.
        labels: 1D array of shape (N,) containing binary labels (0 or 1).
        k_neighbors: Number of nearest neighbors to consider for interpolation.
        target_ratio: Desired ratio of minority / majority class after oversampling.
        seed: Random seed for reproducibility.
        
    Returns:
        A tuple of (augmented_features, augmented_labels).
    """
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int32).reshape(-1)
    
    if x.ndim != 2:
        raise ValueError("Features must be a 2D array.")
    if x.shape[0] != y.shape[0]:
        raise ValueError("Feature and label counts must match.")
        
    unique_classes, counts = np.unique(y, return_counts=True)
    if len(unique_classes) != 2:
        # Cannot perform SMOTE on non-binary datasets
        return x, y
        
    # Determine minority and majority class
    class_0_count, class_1_count = counts[0], counts[1]
    if class_0_count == class_1_count:
        return x, y
        
    min_class = 0 if class_0_count < class_1_count else 1
    maj_class = 1 - min_class
    
    min_features = x[y == min_class]
    maj_features = x[y == maj_class]
    
    n_min = min_features.shape[0]
    n_maj = maj_features.shape[0]
    
    if n_min < 2:
        raise ValueError("SMOTE requires at least 2 samples of the minority class to interpolate.")
        
    # Cap k_neighbors at n_min - 1
    k = min(max(1, int(k_neighbors)), n_min - 1)
    
    # Target count for minority class
    target_min_count = int(round(n_maj * target_ratio))
    n_synthetic = target_min_count - n_min
    
    if n_synthetic <= 0:
        return x, y
        
    # Compute pairwise distances among minority features
    # dists[i, j] is the distance between minority sample i and minority sample j
    diff = min_features[:, None, :] - min_features[None, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # For each sample, sort neighbors (exclude index 0 which is distance to itself)
    neighbors_idx = np.argsort(dists, axis=1)[:, 1 : k + 1]
    
    rng = np.random.default_rng(seed)
    
    # Randomly select minority samples to interpolate from
    base_idx = rng.choice(n_min, size=n_synthetic, replace=True)
    
    # Randomly select which of the k neighbors to interpolate with for each base sample
    neighbor_choice = rng.choice(k, size=n_synthetic, replace=True)
    
    # Interpolation weights
    lambdas = rng.uniform(0.0, 1.0, size=(n_synthetic, 1)).astype(np.float32)
    
    synthetic_samples = []
    for idx, base_i in enumerate(base_idx):
        n_idx = neighbors_idx[base_i, neighbor_choice[idx]]
        diff_vec = min_features[n_idx] - min_features[base_i]
        syn_sample = min_features[base_i] + lambdas[idx] * diff_vec
        synthetic_samples.append(syn_sample)
        
    synthetic_samples = np.stack(synthetic_samples, axis=0)
    synthetic_labels = np.full(n_synthetic, min_class, dtype=np.int32)
    
    # Combine original and synthetic datasets
    augmented_x = np.concatenate([x, synthetic_samples], axis=0)
    augmented_y = np.concatenate([y, synthetic_labels], axis=0)
    
    # Shuffle combined dataset to avoid contiguous blocks
    shuffle_idx = rng.permutation(augmented_x.shape[0])
    return augmented_x[shuffle_idx], augmented_y[shuffle_idx]
