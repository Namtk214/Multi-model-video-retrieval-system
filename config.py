"""Configuration settings for CLIP Features Retrieval System"""

class Config:
    # FAISS settings
    INDEX_TYPE = "flat"  # flat, ivf, hnsw
    
    # Embedding settings
    EMBEDDING_MODEL = "openai/clip-vit-base-patch32"  # 512-dim model compatible with features
    EMBEDDING_DIM = 512  # Updated to match actual CLIP feature dimension
    
    # SuperGlobal reranking parameters
    INITIAL_CANDIDATES_M = 500  # Top-M candidates from FAISS
    QE_NEIGHBORS_R = 10         # Query expansion neighbors
    REFINEMENT_NEIGHBORS_L = 15  # Refinement neighbors
    
    # GeM pooling parameters
    GEM_P_QE = float('inf')     # Max pooling for query expansion
    GEM_P_REFINEMENT = 1.0      # Average pooling for refinement
    
    # Output settings
    TOP_K_RESULTS = 50
    
    # File paths
    INDEX_PATH = "faiss_index.bin"
    EMBEDDINGS_PATH = "embeddings.npy"
    METADATA_PATH = "metadata.json"