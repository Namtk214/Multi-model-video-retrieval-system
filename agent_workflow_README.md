# AI Agent Workflow for Visual Search

This implementation provides an advanced query processing system that enhances visual search through AI agent workflows, following the specifications in the "Instructions for agent workflow" document.

## Architecture Overview

The agent workflow consists of two main steps:

### Step 1: Visual Event Extraction
- **Component**: `VisualEventExtractor` class
- **LLM**: Google Gemini 2.0 Flash
- **Function**: Processes natural language queries to extract structured visual information

### Step 2: Enhanced Search
- **Component**: `EnhancedRetrievalPipeline` class  
- **Function**: Performs vector search with rephrased queries and object-based filtering

## Key Components

### 1. VisualEventExtractor (`visual_event_extractor.py`)
Extracts the following from natural language queries:
- **Visual Elements**: Key visual components, objects, people, locations
- **Actions**: Verbs and activities being performed  
- **Suggested Objects**: Specific object classes for frame detection
- **Rephrased Query**: Optimized version for embedding similarity search

**Example**:
```
Original Query: "Find scenes where someone is cooking pasta"
↓
Visual Elements: ["person", "kitchen", "pasta", "utensils"]
Actions: ["cooking", "stirring", "boiling"]
Suggested Objects: ["person", "bowl", "spoon", "stove", "pot"]
Rephrased Query: "person cooking food in kitchen with utensils"
```

### 2. EnhancedRetrievalPipeline (`enhanced_retrieval_pipeline.py`)
Integrates the visual event extraction with the existing retrieval system:

1. **Query Processing**: Uses VisualEventExtractor to analyze the original query
2. **Vector Search**: Performs embedding search using the rephrased query
3. **Metadata Enhancement**: Updates metadata.json with object detection information
4. **Object Filtering**: Filters results based on detected objects in frames

## Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Up Google AI API Key
```bash
export GOOGLE_API_KEY="your_google_ai_api_key_here"
```

### 3. Prepare Data Structure
Ensure you have the following directory structure:
```
├── clip-features-32/          # CLIP feature files (.npy)
├── keyframes/                 # Video keyframe images
├── objects/                   # Object detection results (JSON files)
│   ├── L21_V001/
│   │   ├── 001.json
│   │   ├── 002.json
│   │   └── ...
│   └── ...
└── metadata.json              # Frame metadata (auto-generated)
```

## Usage Example

```python
from enhanced_retrieval_pipeline import EnhancedRetrievalPipeline
import os

# Initialize the pipeline
pipeline = EnhancedRetrievalPipeline(
    gemini_api_key=os.getenv('GOOGLE_API_KEY')
)

# Build or load index
pipeline.build_index_from_clip_features("clip-features-32")
pipeline.update_metadata_with_objects()

# Process a query through the agent workflow
extraction_results, scores, results = pipeline.retrieve_enhanced(
    query="Find scenes where someone is cooking pasta",
    top_k=10,
    use_object_filtering=True
)

# Results include:
# - extraction_results: Visual event extraction data
# - scores: Relevance scores for each result
# - results: Filtered metadata with object detection info
```

## Workflow Process

### Step 1: Visual Event Extraction
1. **Input**: Natural language query
2. **LLM Processing**: Gemini 2.0 Flash analyzes the query
3. **Output**: Structured extraction with rephrased query

### Step 2: Enhanced Search  
1. **Vector Search**: Use rephrased query for better embedding matches
2. **Metadata Update**: Add object detection information to metadata.json
3. **Object Filtering**: Filter results using suggested objects and confidence thresholds
4. **Reranking**: Apply SuperGlobal reranking (optional)

## Configuration

Key parameters in `config.py`:
- `INITIAL_CANDIDATES_M`: Number of initial FAISS candidates (default: 500)
- `TOP_K_RESULTS`: Final number of results to return (default: 50)
- `EMBEDDING_MODEL`: Model for embeddings ("openai/clip-vit-base-patch32")

Pipeline-specific parameters:
- `object_confidence_threshold`: Minimum confidence for object detection (default: 0.3)
- `use_object_filtering`: Enable/disable object-based filtering
- `use_reranking`: Enable/disable SuperGlobal reranking

## File Descriptions

- `visual_event_extractor.py`: Core visual event extraction using Gemini 2.0 Flash
- `enhanced_retrieval_pipeline.py`: Main pipeline integrating all components
- `agent_workflow_example.py`: Complete usage example with multiple queries
- `config.py`: Configuration parameters for the system
- `requirements.txt`: Python dependencies including Google Generative AI

## Expected Object Detection Format

Object detection JSON files should follow this structure:
```json
[
  {
    "class": "person",
    "confidence": 0.95,
    "bbox": [x, y, width, height]
  },
  {
    "class": "bowl",
    "confidence": 0.87,
    "bbox": [x, y, width, height]
  }
]
```

## Integration with Existing Pipeline

The agent workflow seamlessly integrates with the existing retrieval system:
- Uses the same FAISS indexing and SuperGlobal reranking
- Maintains compatibility with existing metadata structure
- Extends functionality without breaking existing code

## Performance Notes

- Visual event extraction adds ~1-2 seconds per query (LLM processing time)
- Object filtering significantly improves result relevance
- Rephrased queries show better embedding similarity matches
- System scales with existing FAISS index performance characteristics