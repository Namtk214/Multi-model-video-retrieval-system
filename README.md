# 🤖 Multi-Modal Video Retrieval System

An advanced video frame retrieval system that combines **semantic embeddings**, **object detection**, and **OCR text extraction** to provide highly accurate video search capabilities with confidence scoring.

##  System Architecture

![System Architecture](https://github.com/Namtk214/Multi-model-video-retrieval-system/blob/Namtk214-patch-1/pipeline%20project.png)



## Key Features

### Multi-Modal Confidence Scoring
- **60%** Semantic similarity (CLIP embeddings)
- **25%** Object detection confidence matching
- **15%** OCR text similarity matching
- **+5%** Bonus for frames matching both objects AND text

### AI-Powered Query Enhancement
- **Gemini 2.5 Flash** integration for intelligent query processing
- Automatic query rephrasing and object suggestion
- Visual event extraction from natural language queries

### Advanced Retrieval Pipeline
- **SuperGlobal reranking** for improved result quality
- **FAISS indexing** for efficient similarity search
- **Vietnamese & English** language support
- **Semantic object matching** (e.g., "bike" matches "bicycle")

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

### Multi-Modal Weights
The system uses a weighted combination for confidence scoring:
- **Semantic Embeddings**: 60% (primary signal)
- **Object Detection**: 25% (visual objects)
- **OCR Text**: 15% (textual content)
- **Multi-modal Bonus**: +5% when both objects and text match

## 🌐 Supported Languages

### Vietnamese Text Processing
- Accent normalization for better matching
- Vietnamese-specific stop words filtering
- Semantic keyword mapping

### English Text Processing
- Standard NLP preprocessing
- English stop words filtering
- Fuzzy string matching

## 🎯 API Reference

### EnhancedRetrievalPipeline

#### `retrieve_enhanced(query, **kwargs)`
Perform multi-modal video retrieval with confidence scoring.

**Parameters:**
- `query` (str): Natural language search query
- `top_k` (int): Number of results to return (default: 50)
- `use_reranking` (bool): Apply SuperGlobal reranking (default: True)
- `use_object_filtering` (bool): Enable object detection filtering (default: True)
- `use_text_filtering` (bool): Enable OCR text filtering (default: True)
- `object_confidence_threshold` (float): Minimum object confidence (default: 0.3)
- `text_similarity_threshold` (float): Minimum text similarity (default: 0.3)

**Returns:**
- `extraction_results` (dict): AI agent processing results
- `scores` (List[float]): Combined confidence scores
- `results` (List[dict]): Metadata for matching video frames

### VisualEventExtractor

#### `process_query(query)`
Extract visual events and suggested objects from natural language query using Gemini 2.5 Flash.

**Parameters:**
- `query` (str): Input query text

**Returns:**
- Dictionary with rephrased query, visual elements, actions, and suggested objects

### OCRProcessor

#### `extract_text_from_image(image_path)`
Extract text from image using Mistral OCR API.

**Parameters:**
- `image_path` (str): Path to image file

**Returns:**
- Dictionary with extracted text, confidence, and processing metadata

## 📈 Performance

### Benchmarks (on 177K video frames)
- **Index Loading**: ~2 seconds
- **Single Query**: ~3-5 seconds (with reranking)
- **Multi-modal Filtering**: ~1-2 seconds additional
- **Memory Usage**: ~2GB for full index

### Scalability
- Supports millions of video frames
- Efficient FAISS indexing with IVF/HNSW options
- Batch processing for OCR and object detection
- Streaming inference for real-time applications

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

### Adding New Modalities
1. Implement feature extraction pipeline
2. Add filtering method to `EnhancedRetrievalPipeline`
3. Update confidence scoring weights
4. Extend Streamlit UI for new controls

### Custom Object Detection
```python
# Override object detection in metadata
pipeline.update_metadata_with_objects(
    metadata_path="metadata.json",
    objects_base_dir="custom_objects"
)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenAI CLIP** for semantic embeddings
- **Google Gemini 2.5 Flash** for AI-powered query processing
- **Mistral AI** for OCR text extraction
- **FAISS** for efficient similarity search
- **SuperGlobal** for advanced reranking
- **Streamlit** for the interactive web interface

## 📞 Support

For questions and support:
- 📧 Create an issue on GitHub
- 💬 Join our discussion forum
- 📖 Check the documentation wiki

---

⭐ **Star this repository if you find it helpful!**
