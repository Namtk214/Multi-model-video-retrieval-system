# Multi-Modal Video Retrieval System

An advanced video frame retrieval system that combines **semantic embeddings**, **object detection**, and **OCR text extraction** to provide highly accurate video search capabilities with confidence scoring.

Link dataset: 
- https://docs.google.com/spreadsheets/d/1PGE28vdyZVfOBW85PqwY3rcYZVGXEI_wL4a8Ci-c4Gk/edit?gid=0#gid=0
Link query for testing system: 
- https://www.codabench.org/datasets/download/b45400ed-56c7-4576-9a78-ea9eb340b406/
- https://www.codabench.org/datasets/download/5bed0287-eca1-461c-9c03-0a41ff43d0bd/

##  System Architecture

![System Architecture](https://github.com/Namtk214/Multi-model-video-retrieval-system/blob/Namtk214-patch-1/pipeline%20project.png)



## Key Features
### Preprocessing raw videos:
![System Architecture](https://github.com/Namtk214/Multi-model-video-retrieval-system/blob/Namtk214-patch-3/Transnet.png)
- **OpenCV**, for dividing videos into frames.
- **Transnetv2**, using a CNN base architecture for evaluating score for each frames.
- Remove unnecessary frames base on score.




### Multi Modal task: Object detection and Optical Character Recognition (OCR) 
![System Architecture](https://github.com/Namtk214/Multi-model-video-retrieval-system/blob/Namtk214-patch-2/OD%20and%20OCR.png)
- **Faster RCNN**, light weight models for trained in large dataset with enumerous class.
- **Mistral AI OCR**, latest model from mistral for OCR tasks.
- Accuracy approximately 80%.


### Query extraction and rephrasing: Agentic pipeline 
![System Architecture](https://github.com/Namtk214/Multi-model-video-retrieval-system/blob/main/Agent%20pipeline.png)
- **Gemini 2.5 Flash** integration for intelligent query processing
- Automatic query rephrasing and object suggestion
- Visual event extraction from natural language queries

### Embedding model and vector database
![System Architecture](https://github.com/Namtk214/Multi-model-video-retrieval-system/blob/main/Coca%20embedding%20model.png)
- **Coca embedding** using contrastive learning and adding attention layers
- **FAISS** using faiss as vector database for optimizing cosine similarity search process.
- 
### Retrieval pipeline: Reranking method 

**Generalized Mean (GeM) pooling** 
- Computed element-wise across dimensions.  
- The pooled vector is then **L2-normalized**.  

**Special cases:**
-  p = 1  → **Average pooling**  
-  p to infinity → **Max pooling**

---

##  SuperGlobal Steps
1. **Query Expansion (QE)**  
   - Start with the original query vector \( q \) and top-R candidates from FAISS.  
   - Apply **max pooling (p→∞)**:  
2. **Image Descriptor Refinement**  
   - For each candidate \( f_i \), select a neighborhood \( \mathcal{N}_i \) (e.g., L nearest candidates within top-M).  
   - Apply **GeM with p=1 (average pooling)**:  

### Interactive Web Interface
- **Streamlit-based UI** with real-time controls
- Multi-modal scoring breakdown visualization
- Adjustable confidence thresholds
- Detailed result analysis with matched objects and OCR text

##  Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### Required API Keys
Create a `key.env` file with:
```
GOOGLE_API_KEY = 'your_gemini_api_key_here'
MISTRAL_API_KEY = 'your_mistral_api_key_here'
```

### Basic Usage

#### 1. Build Index from CLIP Features
```python
from enhanced_retrieval_pipeline import EnhancedRetrievalPipeline

# Initialize pipeline
pipeline = EnhancedRetrievalPipeline()

# Build index from pre-extracted CLIP features
pipeline.build_index_from_clip_features("path/to/clip-features")

# Save index and metadata
pipeline.save_index()
```

#### 2. Multi-Modal Search
```python
# Load existing index
pipeline.load_index()

# Perform enhanced multi-modal search
extraction_results, scores, results = pipeline.retrieve_enhanced(
    query="weather news report",
    top_k=10,
    use_object_filtering=True,
    use_text_filtering=True,
    object_confidence_threshold=0.3,
    text_similarity_threshold=0.3
)

# Display results with confidence breakdown
for i, result in enumerate(results):
    print(f"Rank {i+1}:")
    print(f"  Combined Score: {result.get('combined_score', 0):.4f}")
    print(f"  Embedding: {result.get('embedding_score', 0):.4f}")
    print(f"  Objects: {result.get('object_score', 0):.3f}")
    print(f"  OCR Text: {result.get('text_score', 0):.3f}")
    print(f"  Multi-modal: {result.get('multi_modal_match', False)}")
```

#### 3. Web Interface
```bash
streamlit run streamlit_app.py
```
Then open http://localhost:8501 in your browser.

## 📊 Data Pipeline

### 1. Extract CLIP Features
```python
from faiss_indexer import FAISSIndexer

# Extract features from video keyframes
indexer = FAISSIndexer()
indexer.extract_and_build_index_from_images("keyframes/")
```

### 2. Add Object Detection Data
```python
# Update metadata with object detection results
pipeline.update_metadata_with_objects(
    metadata_path="metadata.json",
    objects_base_dir="objects"
)
```

### 3. Add OCR Text Data
```python
from update_metadata_with_ocr import update_metadata_with_ocr

# Process images with OCR
update_metadata_with_ocr(
    metadata_path="metadata.json",
    batch_size=10,
    backup_original=True
)
```

## 🔧 Configuration

### Core Settings (`config.py`)
```python
class Config:
    # Embedding Model
    EMBEDDING_MODEL = "openai/clip-vit-large-patch14"
    EMBEDDING_DIM = 768
    
    # FAISS Index
    INDEX_TYPE = "flat"  # "flat", "ivf", or "hnsw"
    
    # Retrieval Parameters
    TOP_K_RESULTS = 50
    INITIAL_CANDIDATES_M = 500
    QE_NEIGHBORS_R = 10
    REFINEMENT_NEIGHBORS_L = 15
    
    # File Paths
    INDEX_PATH = "faiss_index.bin"
    METADATA_PATH = "metadata.json"
```

### Score for each values of model
The system uses a weighted combination for confidence scoring:
- **Semantic Embeddings**: 60% (primary signal)
- **Object Detection**: 25% (visual objects)
- **OCR Text**: 15% (textual content)
- **Multi-modal Bonus**: +5% when both objects and text match








## 🛠️ Development

### Project Structure
```
├── enhanced_retrieval_pipeline.py    # Main multi-modal pipeline
├── visual_event_extractor.py         # AI agent for query processing
├── ocr_processor.py                  # OCR text extraction
├── faiss_indexer.py                  # FAISS vector indexing
├── retrieval_pipeline.py             # Base retrieval pipeline
├── superglobal_reranking.py          # Result reranking
├── streamlit_app.py                  # Web interface
├── update_metadata_with_objects.py   # Object detection integration
├── update_metadata_with_ocr.py       # OCR data integration
├── config.py                         # System configuration
└── requirements.txt                  # Python dependencies
```



## Citation

- **TransNet V2: An effective deep network architecture for fast shot transition detection**, link: https://arxiv.org/pdf/2008.04838
- **A Lightweight Moment Retrieval System with Global Re-Ranking and Robust Adaptive Bidirectional Temporal Search**, link: https://arxiv.org/pdf/2205.01917
- **FAISS**, Meta AI.
- **Gemini 2.5 flash thinking**, Google API Gemini.
- **Mistral AI OCR**, Mistral AI API.



---

⭐ **Star this repository if you find it helpful!**
