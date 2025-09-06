"""Generalized Mean (GeM) pooling implementation"""

import numpy as np
from typing import List, Union
import torch


def gem_pooling(vectors: np.ndarray, p: float = 3.0, eps: float = 1e-6) -> np.ndarray:
    """
    Generalized Mean (GeM) pooling over a set of vectors
    
    GeM_p(X) = (1/n * sum(x_j^p))^(1/p)
    
    Args:
        vectors: Array of shape (N, D) containing N vectors of dimension D
        p: GeM parameter. p=1 gives average pooling, p->inf gives max pooling
        eps: Small epsilon for numerical stability
        
    Returns:
        Pooled vector of shape (D,) that is L2-normalized
    """
    if vectors.ndim != 2:
        raise ValueError("Input vectors must be 2D array of shape (N, D)")
    
    if len(vectors) == 0:
        raise ValueError("Cannot pool empty set of vectors")
    
    # Handle special cases
    if p == 1.0:
        # Average pooling
        pooled = np.mean(vectors, axis=0)
    elif np.isinf(p):
        # Max pooling
        pooled = np.max(vectors, axis=0)
    else:
        # General GeM pooling
        # Add epsilon to avoid numerical issues with small values
        vectors_abs = np.abs(vectors) + eps
        
        # Compute element-wise p-th power
        powered = np.power(vectors_abs, p)
        
        # Take mean across vectors
        mean_powered = np.mean(powered, axis=0)
        
        # Take 1/p-th power
        pooled = np.power(mean_powered, 1.0 / p)
        
        # Restore signs
        signs = np.sign(np.mean(vectors, axis=0))
        pooled = pooled * signs
    
    # L2 normalize
    norm = np.linalg.norm(pooled)
    if norm > eps:
        pooled = pooled / norm
    
    return pooled


def max_pooling(vectors: np.ndarray) -> np.ndarray:
    """
    Max pooling (GeM with p -> infinity)
    
    Args:
        vectors: Array of shape (N, D)
        
    Returns:
        L2-normalized max-pooled vector of shape (D,)
    """
    return gem_pooling(vectors, p=float('inf'))


def average_pooling(vectors: np.ndarray) -> np.ndarray:
    """
    Average pooling (GeM with p = 1)
    
    Args:
        vectors: Array of shape (N, D)
        
    Returns:
        L2-normalized average-pooled vector of shape (D,)
    """
    return gem_pooling(vectors, p=1.0)


class GeMPooling:
    """GeM pooling class for easier parameter management"""
    
    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        self.p = p
        self.eps = eps
    
    def __call__(self, vectors: np.ndarray) -> np.ndarray:
        """Apply GeM pooling to vectors"""
        return gem_pooling(vectors, p=self.p, eps=self.eps)
    
    def pool_with_p(self, vectors: np.ndarray, p: float) -> np.ndarray:
        """Apply GeM pooling with specific p parameter"""
        return gem_pooling(vectors, p=p, eps=self.eps)