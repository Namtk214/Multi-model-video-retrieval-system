"""SuperGlobal reranking implementation with GeM pooling"""

import numpy as np
from typing import List, Dict, Any, Tuple
from gem_pooling import gem_pooling, max_pooling, average_pooling
from config import Config


class SuperGlobalReranker:
    """SuperGlobal reranking with GeM pooling for improved retrieval precision"""
    
    def __init__(self,
                 qe_neighbors_r: int = Config.QE_NEIGHBORS_R,
                 refinement_neighbors_l: int = Config.REFINEMENT_NEIGHBORS_L,
                 gem_p_qe: float = Config.GEM_P_QE,
                 gem_p_refinement: float = Config.GEM_P_REFINEMENT):
        """
        Initialize SuperGlobal reranker
        
        Args:
            qe_neighbors_r: Number of neighbors for query expansion
            refinement_neighbors_l: Number of neighbors for refinement
            gem_p_qe: GeM parameter for query expansion (inf for max pooling)
            gem_p_refinement: GeM parameter for refinement (1.0 for average pooling)
        """
        self.qe_neighbors_r = qe_neighbors_r
        self.refinement_neighbors_l = refinement_neighbors_l
        self.gem_p_qe = gem_p_qe
        self.gem_p_refinement = gem_p_refinement
    
    def rerank(self,
               query_embedding: np.ndarray,
               candidate_embeddings: np.ndarray,
               candidate_metadata: List[Dict[str, Any]],
               top_k: int) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Apply SuperGlobal reranking to candidate embeddings
        
        Args:
            query_embedding: Original query embedding of shape (D,)
            candidate_embeddings: Candidate embeddings of shape (M, D)
            candidate_metadata: Metadata for each candidate
            top_k: Number of final results to return
            
        Returns:
            Tuple of (reranked_scores, reranked_metadata)
        """
        if len(candidate_embeddings) == 0:
            return np.array([]), []
            
        # Ensure query is 1D
        if query_embedding.ndim == 2:
            query_embedding = query_embedding.flatten()
        
        # Step 1: Query Expansion (QE) with p -> inf (max pooling)
        expanded_query = self._query_expansion(query_embedding, candidate_embeddings)
        
        # Step 2: Image Descriptor Refinement with p = 1 (average pooling)
        refined_embeddings = self._descriptor_refinement(candidate_embeddings)
        
        # Step 3: Compute new scores and rerank
        # Calculate individual similarity scores
        query_scores = np.dot(refined_embeddings, query_embedding)
        refined_scores = np.dot(refined_embeddings, expanded_query)
        
        # Final score: (expanded_query + refined_embeddings) / 2
        scores = (query_scores + refined_scores) / 2
        
        # Sort by descending score
        sorted_indices = np.argsort(scores)[::-1]
        
        # Return top-k results
        top_k = min(top_k, len(sorted_indices))
        top_indices = sorted_indices[:top_k]
        
        reranked_scores = scores[top_indices]
        reranked_metadata = [candidate_metadata[i] for i in top_indices]
        
        return reranked_scores, reranked_metadata
    
    def _query_expansion(self, 
                        query_embedding: np.ndarray,
                        candidate_embeddings: np.ndarray) -> np.ndarray:
        """
        Query expansion using top-R candidates and max pooling
        
        Args:
            query_embedding: Original query embedding
            candidate_embeddings: Candidate embeddings
            
        Returns:
            Expanded and normalized query embedding
        """
        # Take top-R candidates for query expansion
        r = min(self.qe_neighbors_r, len(candidate_embeddings))
        
        # Compute similarities to find top-R candidates
        similarities = np.dot(candidate_embeddings, query_embedding)
        top_r_indices = np.argsort(similarities)[::-1][:r]
        top_r_embeddings = candidate_embeddings[top_r_indices]
        
        # Combine query with top-R candidates
        all_vectors = np.vstack([query_embedding.reshape(1, -1), top_r_embeddings])
        
        # Apply GeM pooling with p -> inf (max pooling) for query expansion
        expanded_query = max_pooling(all_vectors)
        
        return expanded_query
    
    def _descriptor_refinement(self, 
                              candidate_embeddings: np.ndarray) -> np.ndarray:
        """
        Refine each candidate descriptor using its neighborhood
        
        Args:
            candidate_embeddings: Candidate embeddings
            
        Returns:
            Refined embeddings
        """
        refined_embeddings = []
        
        for i, candidate in enumerate(candidate_embeddings):
            # Find L nearest neighbors for current candidate
            similarities = np.dot(candidate_embeddings, candidate)
            
            # Get top-L neighbors (including the candidate itself)
            l = min(self.refinement_neighbors_l, len(candidate_embeddings))
            top_l_indices = np.argsort(similarities)[::-1][:l]
            neighborhood = candidate_embeddings[top_l_indices]
            
            # Apply GeM pooling with p = 1 (average pooling) for image descriptor refinement
            refined_embedding = average_pooling(neighborhood)
            
            refined_embeddings.append(refined_embedding)
        
        return np.array(refined_embeddings)
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration"""
        return {
            'qe_neighbors_r': self.qe_neighbors_r,
            'refinement_neighbors_l': self.refinement_neighbors_l,
            'gem_p_qe': self.gem_p_qe,
            'gem_p_refinement': self.gem_p_refinement
        }