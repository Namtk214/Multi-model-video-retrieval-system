"""Main retrieval pipeline combining FAISS indexing, embedding query, and SuperGlobal reranking"""

import numpy as np
from typing import Union, List, Dict, Any, Tuple
import os
import json
from tqdm import tqdm

from faiss_indexer import FAISSIndexer
from embedding_query import EmbeddingQuery
from superglobal_reranking import SuperGlobalReranker
from config import Config


class RetrievalPipeline:
    """Complete GRAB-style retrieval pipeline with SuperGlobal reranking"""
    
    def __init__(self, 
                 model_name: str = Config.EMBEDDING_MODEL,
                 initial_candidates_m: int = Config.INITIAL_CANDIDATES_M,
                 qe_neighbors_r: int = Config.QE_NEIGHBORS_R,
                 refinement_neighbors_l: int = Config.REFINEMENT_NEIGHBORS_L):
        """
        Initialize the complete retrieval pipeline
        
        Args:
            model_name: Hugging Face model name for embeddings
            initial_candidates_m: Number of initial candidates from FAISS
            qe_neighbors_r: Number of neighbors for query expansion
            refinement_neighbors_l: Number of neighbors for refinement
        """
        self.initial_candidates_m = initial_candidates_m
        
        # Initialize components
        self.embedding_query = EmbeddingQuery(model_name)
        self.faiss_indexer = FAISSIndexer(self.embedding_query.get_embedding_dim())
        self.reranker = SuperGlobalReranker(
            qe_neighbors_r=qe_neighbors_r,
            refinement_neighbors_l=refinement_neighbors_l
        )
        
        self.is_indexed = False
        
    def build_index_from_embeddings(self, 
                                  embeddings: np.ndarray,
                                  metadata: List[Dict[str, Any]],
                                  index_type: str = Config.INDEX_TYPE) -> None:
        """
        Build FAISS index from pre-computed embeddings
        
        Args:
            embeddings: Pre-computed embeddings of shape (N, D)
            metadata: List of metadata dicts for each embedding
            index_type: Type of FAISS index to build
        """
        print(f"Building FAISS index with {len(embeddings)} embeddings...")
        self.faiss_indexer.build_index(embeddings, metadata, index_type)
        self.is_indexed = True
        print("Index built successfully!")
        
    
    def build_index_from_clip_features(self,
                                     features_dir: str,
                                     index_type: str = Config.INDEX_TYPE) -> None:
        """
        Build FAISS index from pre-extracted CLIP features directory
        
        Args:
            features_dir: Directory containing .npy CLIP feature files
            index_type: Type of FAISS index to build
        """
        import glob
        
        # Get all .npy files in the features directory
        feature_files = glob.glob(os.path.join(features_dir, "*.npy"))
        if not feature_files:
            raise ValueError(f"No .npy files found in {features_dir}")
            
        print(f"Loading {len(feature_files)} CLIP feature files...")
        embeddings = []
        metadata = []
        
        for feature_file in tqdm(sorted(feature_files), desc="Loading features"):
            try:
                # Load the feature array
                features = np.load(feature_file).astype(np.float32)
                
                # Create metadata from filename
                filename = os.path.basename(feature_file)
                video_id = filename.split('_')[0]  # e.g., L21, L22, etc.
                frame_id = filename.split('_')[1].split('.')[0]  # e.g., V001, V002, etc.
                
                # Handle 2D arrays (num_frames, embedding_dim)
                if features.ndim == 2:
                    for i, feature in enumerate(features):
                        # Normalize the feature vector
                        feature = feature / (np.linalg.norm(feature) + 1e-8)
                        embeddings.append(feature)
                        
                        # Create image path for keyframe - check multiple directories
                        possible_keyframes_dirs = ["keyframes", "keyframes 2", "keyframes 3", "keyframes 4", 
                                                  "keyframes 5", "keyframes 6", "keyframes 7", "keyframes 8", 
                                                  "keyframes 9", "keyframes 10"]
                        image_filename = f"{i+1:03d}.jpg"  # Frame numbers start from 1, zero-padded
                        video_dir = f"{video_id}_{frame_id}"
                        
                        # Find which keyframes directory has this video
                        image_path = None
                        image_exists = False
                        for keyframes_dir in possible_keyframes_dirs:
                            potential_path = f"{keyframes_dir}/{video_dir}/{image_filename}"
                            if os.path.exists(potential_path):
                                image_path = potential_path
                                image_exists = True
                                break
                        
                        # If not found, default to keyframes directory
                        if image_path is None:
                            image_path = f"keyframes/{video_dir}/{image_filename}"
                        
                        metadata.append({
                            'filename': f"{filename}_frame_{i:04d}",
                            'video_id': video_id,
                            'frame_id': frame_id,
                            'frame_index': i,
                            'path': feature_file,
                            'image_path': image_path,
                            'image_exists': image_exists
                        })
                else:
                    # Handle 1D arrays
                    feature = features.flatten()
                    feature = feature / (np.linalg.norm(feature) + 1e-8)
                    embeddings.append(feature)
                    
                    metadata.append({
                        'filename': filename,
                        'video_id': video_id,
                        'frame_id': frame_id,
                        'path': feature_file,
                        'image_path': None,  # No single image for 1D case
                        'image_exists': False
                    })
                
            except Exception as e:
                print(f"Error loading {feature_file}: {e}")
                continue
                
        if not embeddings:
            raise ValueError("No valid feature files could be loaded")
            
        embeddings = np.array(embeddings)
        print(f"Loaded {len(embeddings)} embeddings with dimension {embeddings.shape[1]}")
        
        self.build_index_from_embeddings(embeddings, metadata, index_type)
        
    def retrieve(self, 
                query: str,
                top_k: int = Config.TOP_K_RESULTS,
                use_reranking: bool = True) -> Tuple[List[float], List[Dict[str, Any]]]:
        """
        Retrieve and rerank results for a given text query
        
        Args:
            query: Text string query
            top_k: Number of final results to return
            use_reranking: Whether to apply SuperGlobal reranking
            
        Returns:
            Tuple of (scores, metadata) for top-k results
        """
        if not self.is_indexed:
            raise ValueError("Index not built. Call build_index_from_* first.")
            
        # Step 1: Encode query
        print("Encoding query...")
        query_embedding = self.embedding_query.encode(query)
        if query_embedding.ndim == 2:
            query_embedding = query_embedding[0]  # Remove batch dimension
            
        # Step 2: Initial retrieval from FAISS
        print(f"Retrieving top-{self.initial_candidates_m} candidates from FAISS...")
        initial_scores, initial_indices = self.faiss_indexer.search(
            query_embedding, self.initial_candidates_m
        )
        
        # Get metadata and embeddings for initial candidates
        initial_metadata = self.faiss_indexer.get_metadata(initial_indices)
        
        # Add initial scores to metadata
        for score, metadata in zip(initial_scores, initial_metadata):
            metadata['initial_score'] = float(score)
            
        if not use_reranking:
            # Return initial results without reranking
            top_k = min(top_k, len(initial_metadata))
            return initial_scores[:top_k].tolist(), initial_metadata[:top_k]
            
        # Step 3: Get embeddings for reranking
        print("Preparing embeddings for reranking...")
        # In a real implementation, you might want to store embeddings separately
        # For now, we'll re-encode or assume embeddings are available
        
        # This is a simplified approach - in practice, you'd store embeddings
        candidate_embeddings = []
        for idx in initial_indices:
            # Reconstruct embedding from FAISS index
            # This is approximate for compressed indices
            embedding = self.faiss_indexer.index.reconstruct(int(idx))
            candidate_embeddings.append(embedding)
            
        candidate_embeddings = np.array(candidate_embeddings)
        
        # Step 4: Apply SuperGlobal reranking
        print("Applying SuperGlobal reranking...")
        reranked_scores, reranked_metadata = self.reranker.rerank(
            query_embedding,
            candidate_embeddings,
            initial_metadata,
            top_k
        )
        
        print(f"Retrieved and reranked top-{len(reranked_metadata)} results")
        return reranked_scores.tolist(), reranked_metadata
        
    def save_index(self, 
                  index_path: str = Config.INDEX_PATH,
                  metadata_path: str = Config.METADATA_PATH) -> None:
        """Save FAISS index and metadata to disk"""
        if not self.is_indexed:
            raise ValueError("Index not built. Nothing to save.")
        self.faiss_indexer.save(index_path, metadata_path)
        print(f"Index saved to {index_path}")
        print(f"Metadata saved to {metadata_path}")
        
    def load_index(self,
                  index_path: str = Config.INDEX_PATH,
                  metadata_path: str = Config.METADATA_PATH) -> None:
        """Load FAISS index and metadata from disk"""
        self.faiss_indexer.load(index_path, metadata_path)
        self.is_indexed = True
        print(f"Index loaded from {index_path}")
        print(f"Metadata loaded from {metadata_path}")
        
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the current index"""
        if not self.is_indexed:
            return {"indexed": False}
            
        return {
            "indexed": True,
            "total_embeddings": self.faiss_indexer.index.ntotal,
            "embedding_dim": self.faiss_indexer.embedding_dim,
            "index_type": type(self.faiss_indexer.index).__name__,
            "model_name": self.embedding_query.model_name,
            "reranker_config": self.reranker.get_config()
        }