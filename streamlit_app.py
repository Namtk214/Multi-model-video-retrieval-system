"""Enhanced Streamlit UI for CLIP Features Retrieval System with Agent Workflow"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import re
from enhanced_retrieval_pipeline import EnhancedRetrievalPipeline
from config import Config

# Page configuration
st.set_page_config(
    page_title="Enhanced CLIP Video Retrieval with AI Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f4e79;
        margin-bottom: 2rem;
    }
    .search-box {
        font-size: 1.1rem;
    }
    .result-container {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        background-color: #f9f9f9;
    }
    .result-header {
        font-weight: bold;
        font-size: 1.1rem;
        color: #2c3e50;
    }
    .result-score {
        color: #e74c3c;
        font-weight: bold;
    }
    .result-metadata {
        color: #7f8c8d;
        font-size: 0.9rem;
    }
    .stats-container {
        background-color: #ecf0f1;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def load_api_key():
    """Load Google API key from key.env file"""
    try:
        key_file = "key.env"
        if os.path.exists(key_file):
            with open(key_file, 'r') as f:
                content = f.read()
            pattern = r"GOOGLE_API_KEY\s*=\s*['\"]([^'\"]+)['\"]"
            match = re.search(pattern, content)
            return match.group(1) if match else None
    except Exception:
        pass
    return os.getenv('GOOGLE_API_KEY')

@st.cache_resource
def load_pipeline():
    """Load and cache the enhanced retrieval pipeline"""
    try:
        # Try to load API key
        api_key = load_api_key()
        gemini_available = api_key is not None
        
        # Initialize pipeline
        pipeline = EnhancedRetrievalPipeline(
            gemini_api_key=api_key if gemini_available else None
        )
        
        # Check if index exists
        index_path = Config.INDEX_PATH
        metadata_path = Config.METADATA_PATH
        alternative_index_path = index_path.replace('.bin', '.npy')
        
        if not ((os.path.exists(index_path) or os.path.exists(alternative_index_path)) and os.path.exists(metadata_path)):
            st.error("⚠️ Index not found! Please build the index first using: `python main.py build --features-dir clip-features-32`")
            return None, False
        
        pipeline.load_index(index_path, metadata_path)
        return pipeline, gemini_available
    except Exception as e:
        st.error(f"❌ Error loading pipeline: {str(e)}")
        return None, False

def format_time_from_frame(frame_index, fps=25):
    """Convert frame index to time format (assuming 25 FPS)"""
    seconds = frame_index / fps
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"

def display_results(results_data, search_time, use_reranking, extraction_results=None, use_object_filtering=False, use_text_filtering=False):
    """Display enhanced search results with agent workflow information"""
    if extraction_results:
        extraction_data, scores, results = results_data
    else:
        scores, results = results_data
        extraction_data = None
    
    if not results:
        st.warning("🔍 No results found for your query.")
        return
    
    # Agent Workflow Results (if available)
    if extraction_data:
        st.markdown("### 🤖 AI Agent Processing")
        
        # Create expander for agent details
        with st.expander("🔍 Visual Event Extraction Details", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Original Query:**")
                st.info(extraction_data.get('original_query', 'N/A'))
                
                st.markdown("**Rephrased Query:**")  
                st.success(extraction_data.get('rephrased_query', 'N/A'))
                
            with col2:
                st.markdown("**Visual Elements:**")
                visual_elements = extraction_data.get('visual_elements', [])
                if visual_elements:
                    st.write(", ".join(visual_elements))
                else:
                    st.write("None detected")
                    
                st.markdown("**Actions:**")
                actions = extraction_data.get('actions', [])
                if actions:
                    st.write(", ".join(actions))
                else:
                    st.write("None detected")
                    
                st.markdown("**Suggested Objects:**")
                suggested_objects = extraction_data.get('suggested_objects', [])
                if suggested_objects:
                    st.write(", ".join(suggested_objects))
                else:
                    st.write("None detected")
                    
                st.markdown("**Model Used:**")
                model_used = extraction_data.get('extraction_metadata', {}).get('model_used', 'N/A')
                st.write(model_used)
    
    # Results summary
    st.markdown("### 📊 Search Results")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Results Found", len(results))
    with col2:
        st.metric("Search Time", f"{search_time:.2f}s")
    with col3:
        st.metric("Reranking", "✅ ON" if use_reranking else "❌ OFF")
    with col4:
        st.metric("Object Filter", "✅ ON" if use_object_filtering else "❌ OFF")
    with col5:
        avg_score = np.mean(scores) if scores else 0
        st.metric("Avg Score", f"{avg_score:.3f}")
    
    # Results table
    st.markdown("### 🎯 Top Results")
    
    # Prepare data for display
    display_data = []
    for i, (score, result) in enumerate(zip(scores, results)):
        video_id = result.get('video_id', 'N/A')
        frame_id = result.get('frame_id', 'N/A')
        frame_index = result.get('frame_index', 0)
        initial_score = result.get('initial_score', 'N/A')
        
        # Enhanced scoring information
        embedding_score = result.get('embedding_score', score)
        object_score = result.get('object_score', 0)
        combined_score = result.get('combined_score', score)
        
        # Object information
        matched_objects = result.get('matched_objects', [])
        object_classes = result.get('object_classes', [])
        
        # Estimate timestamp
        timestamp = format_time_from_frame(frame_index)
        
        # Build object info string
        if matched_objects:
            obj_info = ", ".join([f"{obj['class']} ({obj['confidence']:.2f})" for obj in matched_objects[:3]])
        elif object_classes:
            obj_info = ", ".join(object_classes[:3])
        else:
            obj_info = "None"
        
        display_data.append({
            'Rank': i + 1,
            'Video': f"{video_id}_{frame_id}",
            'Frame': frame_index,
            'Timestamp': timestamp,
            'Combined Score': f"{combined_score:.4f}",
            'Embed Score': f"{embedding_score:.4f}",
            'Object Score': f"{object_score:.3f}" if object_score > 0 else "N/A",
            'Matched Objects': obj_info[:50] + "..." if len(obj_info) > 50 else obj_info
        })
    
    # Display as DataFrame
    df = pd.DataFrame(display_data)
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            'Rank': st.column_config.NumberColumn(width="small"),
            'Video': st.column_config.TextColumn(width="medium"),
            'Frame': st.column_config.NumberColumn(width="small"),
            'Timestamp': st.column_config.TextColumn(width="small"),
            'Combined Score': st.column_config.TextColumn(width="medium"),
            'Embed Score': st.column_config.TextColumn(width="medium"),
            'Object Score': st.column_config.TextColumn(width="medium"),
            'Matched Objects': st.column_config.TextColumn(width="large")
        }
    )
    
    # Results cards with inline images
    for i, (score, result) in enumerate(zip(scores[:100], results[:100])):  # Show top 100 results
        # Create a container for each result
        with st.container():
            st.markdown("---")
            
            # Create columns: image on left, details on right
            if result.get('image_exists', False) and result.get('image_path'):
                col1, col2 = st.columns([1, 2])
                
                # Image column
                with col1:
                    try:
                        st.image(result['image_path'], use_column_width=True, 
                                caption=f"Frame {result.get('frame_index', 0)}")
                    except Exception as e:
                        st.error(f"Error loading image: {e}")
                
                # Details column
                with col2:
                    # Rank and score header
                    st.markdown(f"### 🏆 Rank {i+1}")
                    
                    # Enhanced scoring display
                    if result.get('combined_score') is not None:
                        col2a, col2b, col2c = st.columns(3)
                        with col2a:
                            st.metric("Combined Score", f"{result.get('combined_score', score):.4f}")
                        with col2b:
                            st.metric("Embedding", f"{result.get('embedding_score', score):.4f}")
                        with col2c:
                            obj_score = result.get('object_score', 0)
                            st.metric("Object Match", f"{obj_score:.3f}" if obj_score > 0 else "N/A")
                    else:
                        col2a, col2b = st.columns(2)
                        with col2a:
                            st.metric("Final Score", f"{score:.4f}")
                        with col2b:
                            if 'initial_score' in result:
                                improvement = ((score / result['initial_score'] - 1) * 100) if result['initial_score'] > 0 else 0
                                st.metric("Improvement", f"+{improvement:.1f}%", f"from {result['initial_score']:.4f}")
                    
                    # Video information
                    st.markdown(f"**🎬 Video:** {result.get('video_id', 'N/A')}_{result.get('frame_id', 'N/A')}")
                    st.markdown(f"**🎞️ Frame:** {result.get('frame_index', 'N/A')} | **⏰ Time:** {format_time_from_frame(result.get('frame_index', 0))}")
                    st.markdown(f"**📁 File:** `{result.get('filename', 'N/A')}`")
                    
                    # Object information (if available)
                    matched_objects = result.get('matched_objects', [])
                    if matched_objects:
                        st.markdown("**🎯 Matched Objects:**")
                        for obj in matched_objects[:5]:
                            st.markdown(f"• {obj['class']} (confidence: {obj['confidence']:.2f})")
                    elif result.get('object_classes'):
                        st.markdown("**👁️ Detected Objects:**")
                        obj_classes = result.get('object_classes', [])[:8]
                        st.markdown(f"• {', '.join(obj_classes)}")
                    
                    # Expandable additional details
                    with st.expander("📋 More Details"):
                        st.markdown(f"**📂 Feature Path:** `{result.get('path', 'N/A')}`")
                        st.markdown(f"**🖼️ Image Path:** `{result.get('image_path', 'N/A')}`")
                        
                        # Extended object information
                        if result.get('has_object_detection', False):
                            st.markdown(f"**🔢 Total Objects:** {result.get('num_detected_objects', 0)}")
                            all_objects = result.get('object_classes', [])
                            if len(all_objects) > 8:
                                st.markdown(f"**All Objects:** {', '.join(all_objects)}")
                        else:
                            st.markdown("**Object Detection:** Not available")
            else:
                # No image available - single column layout
                st.markdown(f"### 🏆 Rank {i+1} - {result.get('video_id', 'N/A')}_{result.get('frame_id', 'N/A')}")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Final Score", f"{score:.4f}")
                with col2:
                    if 'initial_score' in result:
                        improvement = ((score / result['initial_score'] - 1) * 100) if result['initial_score'] > 0 else 0
                        st.metric("Improvement", f"+{improvement:.1f}%")
                with col3:
                    st.metric("Frame", result.get('frame_index', 'N/A'))
                
                st.markdown(f"**🎞️ Time:** {format_time_from_frame(result.get('frame_index', 0))} | **📁 File:** `{result.get('filename', 'N/A')}`")
                st.info("🖼️ No keyframe image available for this result")

def main():
    # Header
    st.markdown('<h1 class="main-header">🤖 Enhanced CLIP Video Retrieval with AI Agent</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load pipeline
    with st.spinner("🔄 Loading enhanced retrieval system..."):
        pipeline, gemini_available = load_pipeline()
    
    if pipeline is None:
        st.stop()
    
    # API Status
    if gemini_available:
        st.success("✅ Gemini 2.0 Flash API available - Full AI Agent features enabled")
    else:
        st.warning("⚠️ Gemini API not available - Using fallback mode (add API key to key.env file)")
    
    # Display system info
    stats = pipeline.get_stats()
    with st.expander("ℹ️ Enhanced System Information"):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Embeddings", f"{stats.get('total_embeddings', 0):,}")
        with col2:
            st.metric("Embedding Dimension", stats.get('embedding_dim', 0))
        with col3:
            st.metric("Index Type", stats.get('index_type', 'Unknown'))
        with col4:
            obj_detection = "✅ Available" if stats.get('has_object_detection', False) else "❌ Not Available"
            st.metric("Object Detection", obj_detection)
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Enhanced Search Configuration")
    
    # AI Agent Settings
    st.sidebar.subheader("🤖 AI Agent Workflow")
    use_agent_workflow = st.sidebar.checkbox("🧠 Enable AI Agent Processing", value=gemini_available, disabled=not gemini_available)
    
    if use_agent_workflow:
        use_object_filtering = st.sidebar.checkbox("🎯 Object-based Filtering", value=True)
        use_text_filtering = st.sidebar.checkbox("📝 OCR Text Filtering", value=True)
        
        if use_object_filtering:
            object_confidence = st.sidebar.slider("Object Confidence Threshold", 
                                                 min_value=0.1, max_value=0.9, value=0.3, step=0.1)
            match_threshold = st.sidebar.slider("Object Match Threshold", 
                                               min_value=0.2, max_value=0.8, value=0.4, step=0.1)
        else:
            object_confidence = 0.3
            match_threshold = 0.4
        
        if use_text_filtering:
            text_similarity = st.sidebar.slider("Text Similarity Threshold", 
                                               min_value=0.1, max_value=0.8, value=0.3, step=0.1)
        else:
            text_similarity = 0.3
    else:
        use_object_filtering = False
        use_text_filtering = False
        object_confidence = 0.3
        match_threshold = 0.4
        text_similarity = 0.3
    
    st.sidebar.markdown("---")
    
    # Search parameters
    top_k = st.sidebar.slider("🔢 Number of Results", min_value=5, max_value=200, value=50, step=5)
    use_reranking = st.sidebar.checkbox("🔄 Enable SuperGlobal Reranking", value=True)
    
    # Advanced parameters (collapsed by default)
    with st.sidebar.expander("🔧 Advanced Settings"):
        initial_candidates = st.slider("Initial Candidates", min_value=100, max_value=1000, value=500, step=50)
        qe_neighbors = st.slider("QE Neighbors", min_value=5, max_value=30, value=10, step=1)
        refinement_neighbors = st.slider("Refinement Neighbors", min_value=5, max_value=30, value=15, step=1)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📖 Enhanced Usage Tips")
    st.sidebar.markdown("""
    **🤖 AI Agent Mode:**
    - Processes natural language queries
    - Automatically extracts visual elements
    - Filters by detected objects
    
    **🔍 Example Queries:**
    - "bike racers competing"
    - "people eating spring rolls" 
    - "news anchor reporting"
    - "bản tin thời sự"
    - "children playing soccer"
    
    **💡 Tips:**
    - Enable AI Agent for best results
    - Object filtering improves precision
    - Works with Vietnamese & English
    """)
    
    # Main search interface
    st.markdown("### 🔍 Search Videos")
    
    # Search input
    query = st.text_input(
        "Enter your search query:",
        placeholder="E.g., 'bike racers competing' or 'people eating spring rolls' or 'bản tin thời sự'",
        key="search_query"
    )
    
    # Search button
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        search_button = st.button("🚀 Search", type="primary", use_container_width=True)
    
    # Perform search
    if search_button and query.strip():
        search_spinner_text = "🤖 Processing with AI Agent..." if use_agent_workflow else "🔍 Searching..."
        
        with st.spinner(search_spinner_text):
            try:
                # Update pipeline parameters if needed
                if hasattr(pipeline, 'base_pipeline'):
                    pipeline.base_pipeline.initial_candidates_m = initial_candidates
                    pipeline.base_pipeline.reranker.qe_neighbors_r = qe_neighbors
                    pipeline.base_pipeline.reranker.refinement_neighbors_l = refinement_neighbors
                
                # Perform search based on mode
                start_time = time.time()
                
                if use_agent_workflow and gemini_available:
                    # Enhanced search with AI agent workflow
                    extraction_results, scores, results = pipeline.retrieve_enhanced(
                        query=query.strip(),
                        top_k=top_k,
                        use_reranking=use_reranking,
                        use_object_filtering=use_object_filtering,
                        object_confidence_threshold=object_confidence
                    )
                    search_time = time.time() - start_time
                    
                    # Display enhanced results
                    display_results(
                        (extraction_results, scores, results), 
                        search_time, 
                        use_reranking, 
                        extraction_results=True,
                        use_object_filtering=use_object_filtering,
                        use_text_filtering=use_text_filtering
                    )
                    
                else:
                    # Standard search
                    scores, results = pipeline.base_pipeline.retrieve(
                        query.strip(),
                        top_k=top_k,
                        use_reranking=use_reranking
                    )
                    search_time = time.time() - start_time
                    
                    # Display standard results
                    display_results(
                        (scores, results), 
                        search_time, 
                        use_reranking,
                        extraction_results=False,
                        use_object_filtering=False,
                        use_text_filtering=False
                    )
                
            except Exception as e:
                st.error(f"❌ Search failed: {str(e)}")
                st.exception(e)
    
    elif search_button:
        st.warning("⚠️ Please enter a search query.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #7f8c8d; font-size: 0.9rem;'>
        🤖 Enhanced CLIP Video Retrieval with AI Agent Workflow<br>
        Built with Streamlit • Powered by CLIP, FAISS, SuperGlobal Reranking & Gemini 2.0 Flash
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()