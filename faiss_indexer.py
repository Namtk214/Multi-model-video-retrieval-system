"""FAISS indexing for keyframe embeddings"""

import numpy as np
import json
import os
from typing import List, Tuple, Optional
from config import Config

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    from simple_indexer import SimpleIndexer


class FAISSIndexer:
    def __init__(self, embedding_dim: int = Config.EMBEDDING_DIM):
        self.embedding_dim = embedding_dim
        self.index = None
        self.metadata = []
        self.use_simple_indexer = not FAISS_AVAILABLE
        if not FAISS_AVAILABLE:
            print("Warning: FAISS not available, using simple brute-force indexer")
        
    def build_index(self, embeddings: np.ndarray, metadata: List[dict], 
                   index_type: str = Config.INDEX_TYPE) -> None:
        """
        Build FAISS index from embeddings
        
        Args:
            embeddings: Array of shape (N, D) with L2-normalized embeddings
            metadata: List of metadata dicts for each embedding
            index_type: Type of FAISS index ('flat', 'ivf', 'hnsw')
        """
        if self.use_simple_indexer:
            self.index = SimpleIndexer(self.embedding_dim)
            self.index.build_index(embeddings, metadata, index_type)
            self.metadata = metadata
            return
            
        embeddings = embeddings.astype(np.float32)
        
        # Ensure embeddings are L2-normalized
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-8)
        
        if index_type == "flat":
            # Use IndexFlatIP for cosine similarity with normalized vectors
            self.index = faiss.IndexFlatIP(self.embedding_dim)
        elif index_type == "ivf":
            # IVF-PQ for large datasets
            nlist = min(100, len(embeddings) // 10)  # Number of clusters
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            self.index = faiss.IndexIVFPQ(quantizer, self.embedding_dim, nlist, 64, 8)
            self.index.train(embeddings)
        elif index_type == "hnsw":
            # HNSW for fast approximate search
            self.index = faiss.IndexHNSWFlat(self.embedding_dim, 32)
            self.index.hnsw.efConstruction = 40
        else:
            raise ValueError(f"Unknown index type: {index_type}")
            
        self.index.add(embeddings)
        self.metadata = metadata
        
    def search(self, query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for top-k similar embeddings
        
        Args:
            query: Query embedding of shape (1, D) or (D,)
            k: Number of results to return
            
        Returns:
            scores: Similarity scores
            indices: Indices of retrieved embeddings
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index first.")
        
        if self.use_simple_indexer:
            return self.index.search(query, k)
            
        if query.ndim == 1:
            query = query.reshape(1, -1)
            
        query = query.astype(np.float32)
        
        # Ensure query is L2-normalized
        norm = np.linalg.norm(query, axis=1, keepdims=True)
        query = query / (norm + 1e-8)
        
        scores, indices = self.index.search(query, k)
        return scores[0], indices[0]
    
    def get_metadata(self, indices: np.ndarray) -> List[dict]:
        """Get metadata for given indices"""
        return [self.metadata[i] for i in indices if i < len(self.metadata)]
    
    def save(self, index_path: str = Config.INDEX_PATH, 
             metadata_path: str = Config.METADATA_PATH) -> None:
        """Save index and metadata to disk"""
        if self.index is None:
            raise ValueError("Index not built. Call build_index first.")
        
        if self.use_simple_indexer:
            self.index.save(index_path, metadata_path)
            return
            
        faiss.write_index(self.index, index_path)
        
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def load(self, index_path: str = Config.INDEX_PATH,
             metadata_path: str = Config.METADATA_PATH) -> None:
        """Load index and metadata from disk"""
        if self.use_simple_indexer:
            self.index = SimpleIndexer(self.embedding_dim)
            self.index.load(index_path, metadata_path)
            self.metadata = self.index.metadata
            return
            
        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            raise FileNotFoundError("Index or metadata file not found")
            
        self.index = faiss.read_index(index_path)
        
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)