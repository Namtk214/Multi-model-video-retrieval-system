"""Simple alternative to FAISS for testing when FAISS is not available"""

import numpy as np
import json
import os
from typing import List, Dict, Any, Tuple
from config import Config


class SimpleIndexer:
    """Simple brute-force indexer as FAISS alternative for testing"""
    
    def __init__(self, embedding_dim: int = Config.EMBEDDING_DIM):
        self.embedding_dim = embedding_dim
        self.embeddings = None
        self.metadata = []
        self.ntotal = 0
        
    def build_index(self, embeddings: np.ndarray, metadata: List[dict], 
                   index_type: str = Config.INDEX_TYPE) -> None:
        """
        Build simple index from embeddings
        
        Args:
            embeddings: Array of shape (N, D) with L2-normalized embeddings
            metadata: List of metadata dicts for each embedding
            index_type: Ignored in simple implementation
        """
        embeddings = embeddings.astype(np.float32)
        
        # Ensure embeddings are L2-normalized
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-8)
        
        self.embeddings = embeddings
        self.metadata = metadata
        self.ntotal = len(embeddings)
        
    def search(self, query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for top-k similar embeddings using cosine similarity
        
        Args:
            query: Query embedding of shape (1, D) or (D,)
            k: Number of results to return
            
        Returns:
            scores: Similarity scores
            indices: Indices of retrieved embeddings
        """
        if self.embeddings is None:
            raise ValueError("Index not built. Call build_index first.")
            
        if query.ndim == 1:
            query = query.reshape(1, -1)
            
        query = query.astype(np.float32)
        
        # Ensure query is L2-normalized
        norm = np.linalg.norm(query, axis=1, keepdims=True)
        query = query / (norm + 1e-8)
        
        # Compute cosine similarities (dot product for normalized vectors)
        similarities = np.dot(self.embeddings, query.T).flatten()
        
        # Get top-k results
        k = min(k, len(similarities))
        top_k_indices = np.argpartition(similarities, -k)[-k:]
        top_k_indices = top_k_indices[np.argsort(-similarities[top_k_indices])]
        
        scores = similarities[top_k_indices]
        
        return scores, top_k_indices
    
    def get_metadata(self, indices: np.ndarray) -> List[dict]:
        """Get metadata for given indices"""
        return [self.metadata[i] for i in indices if i < len(self.metadata)]
    
    def reconstruct(self, idx: int) -> np.ndarray:
        """Reconstruct embedding at given index"""
        if self.embeddings is None:
            raise ValueError("Index not built.")
        return self.embeddings[idx]
    
    def add(self, embeddings: np.ndarray) -> None:
        """Add embeddings to index (for compatibility)"""
        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])
        self.ntotal = len(self.embeddings)
    
    def save(self, index_path: str = Config.INDEX_PATH, 
             metadata_path: str = Config.METADATA_PATH) -> None:
        """Save index and metadata to disk"""
        if self.embeddings is None:
            raise ValueError("Index not built. Call build_index first.")
            
        # Save embeddings as numpy array
        np.save(index_path.replace('.bin', '.npy'), self.embeddings)
        
        # Save metadata as JSON
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def load(self, index_path: str = Config.INDEX_PATH,
             metadata_path: str = Config.METADATA_PATH) -> None:
        """Load index and metadata from disk"""
        embeddings_path = index_path.replace('.bin', '.npy')
        
        if not os.path.exists(embeddings_path) or not os.path.exists(metadata_path):
            # Try to find the files
            possible_paths = [
                embeddings_path,
                index_path + '.npy',
                'faiss_index.npy'
            ]
            found_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    found_path = path
                    break
            
            if found_path is None or not os.path.exists(metadata_path):
                available_files = [f for f in os.listdir('.') if f.endswith(('.npy', '.json'))]
                raise FileNotFoundError(f"Index or metadata file not found. Looking for {embeddings_path} and {metadata_path}. Available files: {available_files}")
            
            embeddings_path = found_path
            
        self.embeddings = np.load(embeddings_path)
        self.ntotal = len(self.embeddings)
        
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)