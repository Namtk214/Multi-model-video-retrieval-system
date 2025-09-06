"""Enhanced Retrieval Pipeline with Visual Event Extraction and Object-based Filtering"""

import numpy as np
import json
import os
from typing import Union, List, Dict, Any, Tuple, Optional
from tqdm import tqdm

from visual_event_extractor import VisualEventExtractor
from retrieval_pipeline import RetrievalPipeline
from ocr_processor import OCRProcessor, load_mistral_api_key
from config import Config


class EnhancedRetrievalPipeline:
    """Enhanced retrieval pipeline with visual event extraction and object-based filtering"""
    
    def __init__(self, 
                 gemini_api_key: Optional[str] = None,
                 mistral_api_key: Optional[str] = None,
                 model_name: str = Config.EMBEDDING_MODEL,
                 initial_candidates_m: int = Config.INITIAL_CANDIDATES_M,
                 qe_neighbors_r: int = Config.QE_NEIGHBORS_R,
                 refinement_neighbors_l: int = Config.REFINEMENT_NEIGHBORS_L):
        """
        Initialize the enhanced retrieval pipeline
        
        Args:
            gemini_api_key: Google AI API key for Gemini 2.5 Flash
            mistral_api_key: Mistral API key for OCR processing
            model_name: Hugging Face model name for embeddings
            initial_candidates_m: Number of initial candidates from FAISS
            qe_neighbors_r: Number of neighbors for query expansion
            refinement_neighbors_l: Number of neighbors for refinement
        """
        # Initialize visual event extractor
        self.visual_extractor = VisualEventExtractor(api_key=gemini_api_key)
        
        # Initialize OCR processor
        ocr_key = mistral_api_key or load_mistral_api_key()
        self.ocr_processor = OCRProcessor(api_key=ocr_key)
        
        # Initialize base retrieval pipeline
        self.base_pipeline = RetrievalPipeline(
            model_name=model_name,
            initial_candidates_m=initial_candidates_m,
            qe_neighbors_r=qe_neighbors_r,
            refinement_neighbors_l=refinement_neighbors_l
        )
        
        # Object detection results cache
        self.object_detection_cache = {}
        
    def build_index_from_embeddings(self, 
                                  embeddings: np.ndarray,
                                  metadata: List[Dict[str, Any]],
                                  index_type: str = Config.INDEX_TYPE) -> None:
        """Build FAISS index from pre-computed embeddings"""
        self.base_pipeline.build_index_from_embeddings(embeddings, metadata, index_type)
        
    def build_index_from_clip_features(self,
                                     features_dir: str,
                                     index_type: str = Config.INDEX_TYPE) -> None:
        """Build FAISS index from pre-extracted CLIP features directory"""
        self.base_pipeline.build_index_from_clip_features(features_dir, index_type)
        
    def update_metadata_with_objects(self, 
                                   metadata_path: str = Config.METADATA_PATH,
                                   objects_base_dir: str = "objects") -> None:
        """
        Update metadata.json with object detection information for each frame
        
        Args:
            metadata_path: Path to the metadata.json file
            objects_base_dir: Base directory containing object detection results
        """
        print("Updating metadata with object detection information...")
        
        # Load existing metadata
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata_list = json.load(f)
        else:
            # Get metadata from FAISS indexer
            metadata_list = []
            if hasattr(self.base_pipeline.faiss_indexer, 'metadata'):
                metadata_list = self.base_pipeline.faiss_indexer.metadata
        
        updated_count = 0
        
        for metadata in tqdm(metadata_list, desc="Processing metadata"):
            # Extract video and frame information
            video_id = metadata.get('video_id', '')
            frame_id = metadata.get('frame_id', '')
            frame_index = metadata.get('frame_index', 0)
            
            if not video_id or not frame_id:
                continue
                
            # Construct object detection file path
            video_dir = f"{video_id}_{frame_id}"
            frame_filename = f"{frame_index + 1:03d}.json"  # Frame numbers start from 1
            
            # Look for object detection results in multiple possible directories
            possible_object_dirs = [
                f"{objects_base_dir}/{video_dir}",
                f"objects/{video_dir}",
                f"object_detection/{video_dir}",
                f"detections/{video_dir}"
            ]
            
            object_file_path = None
            for obj_dir in possible_object_dirs:
                potential_path = os.path.join(obj_dir, frame_filename)
                if os.path.exists(potential_path):
                    object_file_path = potential_path
                    break
            
            # Load object detection results if available
            if object_file_path and os.path.exists(object_file_path):
                try:
                    with open(object_file_path, 'r') as f:
                        object_data = json.load(f)
                    
                    # Extract object classes and confidence scores
                    detected_objects = []
                    if isinstance(object_data, list):
                        for detection in object_data:
                            if isinstance(detection, dict) and 'class' in detection:
                                detected_objects.append({
                                    'class': detection['class'],
                                    'confidence': detection.get('confidence', 0.0),
                                    'bbox': detection.get('bbox', [])
                                })
                    elif isinstance(object_data, dict) and 'detections' in object_data:
                        for detection in object_data['detections']:
                            if isinstance(detection, dict) and 'class' in detection:
                                detected_objects.append({
                                    'class': detection['class'],
                                    'confidence': detection.get('confidence', 0.0),
                                    'bbox': detection.get('bbox', [])
                                })
                    
                    # Update metadata
                    metadata['detected_objects'] = detected_objects
                    metadata['object_detection_file'] = object_file_path
                    metadata['has_object_detection'] = True
                    updated_count += 1
                    
                except Exception as e:
                    print(f"Error loading object detection for {object_file_path}: {e}")
                    metadata['detected_objects'] = []
                    metadata['has_object_detection'] = False
            else:
                metadata['detected_objects'] = []
                metadata['has_object_detection'] = False
        
        # Save updated metadata
        with open(metadata_path, 'w') as f:
            json.dump(metadata_list, f, indent=2)
            
        print(f"Updated {updated_count}/{len(metadata_list)} metadata entries with object detection")
        
    def filter_by_objects(self, 
                         results_metadata: List[Dict[str, Any]], 
                         suggested_objects: List[str],
                         confidence_threshold: float = 0.3,
                         match_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Enhanced filter search results based on detected objects in frames
        
        Args:
            results_metadata: List of metadata from search results
            suggested_objects: List of object classes to filter by
            confidence_threshold: Minimum confidence for object detection (already filtered in metadata)
            match_threshold: Minimum object matching score to consider frame relevant
            
        Returns:
            Filtered list of metadata with object matching scores
        """
        if not suggested_objects:
            return results_metadata
            
        filtered_results = []
        
        # Normalize suggested objects for better matching
        normalized_suggested = [obj.lower().strip() for obj in suggested_objects]
        
        for metadata in results_metadata:
            detected_objects = metadata.get('detected_objects', [])
            object_classes = metadata.get('object_classes', [])
            
            if not detected_objects:
                metadata['object_filter_matched'] = False
                metadata['object_match_score'] = 0.0
                metadata['matched_objects'] = []
                continue
            
            # Calculate matching score
            matched_objects = []
            total_confidence = 0.0
            
            # Fast check using object_classes first
            quick_matches = [obj for obj in object_classes 
                           if any(suggested.lower() in obj.lower() or obj.lower() in suggested.lower() 
                                 for suggested in normalized_suggested)]
            
            if quick_matches:
                # Detailed matching with confidence scores
                for detection in detected_objects:
                    obj_class = detection.get('class', '').lower()
                    confidence = detection.get('confidence', 0.0)
                    
                    # Check for exact or partial matches
                    for suggested in normalized_suggested:
                        if (suggested in obj_class or obj_class in suggested or 
                            self._semantic_match(obj_class, suggested)):
                            matched_objects.append({
                                'class': detection.get('class', ''),
                                'confidence': confidence,
                                'suggested_match': suggested
                            })
                            total_confidence += confidence
                            break
                
                # Calculate match score (average confidence of matched objects)
                match_score = total_confidence / len(matched_objects) if matched_objects else 0.0
                
                # Apply threshold
                if match_score >= match_threshold:
                    metadata['object_filter_matched'] = True
                    metadata['object_match_score'] = match_score
                    metadata['matched_objects'] = matched_objects
                    filtered_results.append(metadata)
                else:
                    metadata['object_filter_matched'] = False
                    metadata['object_match_score'] = match_score
                    metadata['matched_objects'] = matched_objects
            else:
                metadata['object_filter_matched'] = False
                metadata['object_match_score'] = 0.0
                metadata['matched_objects'] = []
                
        # Sort by object match score (descending)
        filtered_results.sort(key=lambda x: x.get('object_match_score', 0), reverse=True)
        
        return filtered_results
    
    def _semantic_match(self, detected_class: str, suggested_class: str) -> bool:
        """
        Check for semantic matches between detected and suggested object classes
        
        Args:
            detected_class: Class from object detection
            suggested_class: Class from query suggestion
            
        Returns:
            True if semantic match found
        """
        # Common semantic mappings
        semantic_mappings = {
            'bicycle': ['bike', 'cycle', 'cycling'],
            'bike': ['bicycle', 'cycle', 'cycling'],
            'motorcycle': ['bike', 'motorbike', 'scooter'],
            'car': ['vehicle', 'automobile', 'auto'],
            'vehicle': ['car', 'automobile', 'truck', 'bus'],
            'person': ['human', 'people', 'man', 'woman', 'boy', 'girl'],
            'human': ['person', 'people', 'man', 'woman'],
            'water': ['rain', 'flood', 'river', 'lake'],
            'rain': ['water', 'weather', 'storm'],
            'street': ['road', 'highway', 'path'],
            'building': ['house', 'structure', 'architecture'],
            'food': ['meal', 'dish', 'cooking', 'kitchen']
        }
        
        # Check if detected class has semantic mappings that match suggested
        if detected_class in semantic_mappings:
            return any(match in suggested_class for match in semantic_mappings[detected_class])
        
        # Check reverse mapping
        if suggested_class in semantic_mappings:
            return any(match in detected_class for match in semantic_mappings[suggested_class])
        
        return False
    
    def filter_by_ocr_text(self, 
                          results_metadata: List[Dict[str, Any]], 
                          query_text: str,
                          text_similarity_threshold: float = 0.3) -> List[Dict[str, Any]]:
        """
        Filter search results based on OCR text content in frames
        
        Args:
            results_metadata: List of metadata from search results
            query_text: Text query to match against OCR content
            text_similarity_threshold: Minimum text similarity score to include frame
            
        Returns:
            Filtered list of metadata with text matching scores
        """
        if not query_text.strip():
            return results_metadata
        
        filtered_results = []
        query_words = set(query_text.lower().split())
        
        for metadata in results_metadata:
            ocr_text = metadata.get('ocr_text', '').lower()
            ocr_keywords = metadata.get('ocr_keywords', [])
            
            if not ocr_text and not ocr_keywords:
                metadata['text_filter_matched'] = False
                metadata['text_match_score'] = 0.0
                metadata['matched_text_keywords'] = []
                continue
            
            # Calculate text similarity score
            text_match_score = 0.0
            matched_keywords = []
            
            # Check for exact word matches in OCR text
            ocr_words = set(ocr_text.split())
            word_matches = query_words.intersection(ocr_words)
            
            if word_matches:
                text_match_score += len(word_matches) / len(query_words) * 0.8
                matched_keywords.extend(list(word_matches))
            
            # Check for partial matches in OCR keywords
            for keyword in ocr_keywords:
                keyword_lower = keyword.lower()
                for query_word in query_words:
                    if (query_word in keyword_lower or keyword_lower in query_word or
                        self._fuzzy_match(query_word, keyword_lower)):
                        text_match_score += 0.1
                        matched_keywords.append(keyword)
                        break
            
            # Check for phrase matches (substring search)
            query_lower = query_text.lower()
            if query_lower in ocr_text:
                text_match_score += 0.5
            
            # Vietnamese text matching
            text_match_score += self._vietnamese_text_similarity(query_text, ocr_text)
            
            # Apply threshold
            text_match_score = min(text_match_score, 1.0)  # Cap at 1.0
            
            if text_match_score >= text_similarity_threshold:
                metadata['text_filter_matched'] = True
                metadata['text_match_score'] = text_match_score
                metadata['matched_text_keywords'] = list(set(matched_keywords))
                filtered_results.append(metadata)
            else:
                metadata['text_filter_matched'] = False
                metadata['text_match_score'] = text_match_score
                metadata['matched_text_keywords'] = list(set(matched_keywords))
        
        # Sort by text match score (descending)
        filtered_results.sort(key=lambda x: x.get('text_match_score', 0), reverse=True)
        
        return filtered_results
    
    def _fuzzy_match(self, word1: str, word2: str, threshold: float = 0.8) -> bool:
        """
        Simple fuzzy string matching using character overlap
        
        Args:
            word1, word2: Words to compare
            threshold: Minimum similarity threshold
            
        Returns:
            True if words are similar enough
        """
        if len(word1) < 3 or len(word2) < 3:
            return word1 == word2
        
        # Calculate character overlap
        chars1 = set(word1)
        chars2 = set(word2)
        overlap = len(chars1.intersection(chars2))
        union = len(chars1.union(chars2))
        
        similarity = overlap / union if union > 0 else 0
        return similarity >= threshold
    
    def _vietnamese_text_similarity(self, query: str, ocr_text: str) -> float:
        """
        Calculate similarity for Vietnamese text with accent normalization
        
        Args:
            query: Query text
            ocr_text: OCR extracted text
            
        Returns:
            Similarity score between 0 and 1
        """
        # Normalize Vietnamese characters for better matching
        import unicodedata
        
        def normalize_vietnamese(text):
            # Remove Vietnamese accents for fuzzy matching
            text = unicodedata.normalize('NFD', text)
            text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
            return text.lower()
        
        normalized_query = normalize_vietnamese(query)
        normalized_ocr = normalize_vietnamese(ocr_text)
        
        # Check for substring matches in normalized text
        query_words = normalized_query.split()
        match_count = 0
        
        for word in query_words:
            if len(word) >= 3 and word in normalized_ocr:
                match_count += 1
        
        return (match_count / len(query_words)) * 0.3 if query_words else 0
        
    def retrieve_enhanced(self, 
                         query: str,
                         top_k: int = Config.TOP_K_RESULTS,
                         use_reranking: bool = True,
                         use_object_filtering: bool = True,
                         use_text_filtering: bool = True,
                         object_confidence_threshold: float = 0.3,
                         text_similarity_threshold: float = 0.3) -> Tuple[Dict[str, Any], List[float], List[Dict[str, Any]]]:
        """
        Enhanced retrieval with visual event extraction, object-based filtering, and OCR text filtering
        
        Args:
            query: Original natural language query
            top_k: Number of final results to return
            use_reranking: Whether to apply SuperGlobal reranking
            use_object_filtering: Whether to filter by detected objects
            use_text_filtering: Whether to filter by OCR text content
            object_confidence_threshold: Minimum confidence for object detection
            text_similarity_threshold: Minimum text similarity for OCR filtering
            
        Returns:
            Tuple of (extraction_results, scores, filtered_metadata)
        """
        # Step 1: Visual Event Extraction
        print("Step 1: Extracting visual events from query...")
        extraction_results = self.visual_extractor.process_query(query)
        
        rephrased_query = extraction_results["rephrased_query"]
        suggested_objects = extraction_results["suggested_objects"]
        
        print(f"Original query: {query}")
        print(f"Rephrased query: {rephrased_query}")
        print(f"Suggested objects: {suggested_objects}")
        
        # Step 2: Enhanced Search with rephrased query
        print("Step 2: Performing enhanced search...")
        scores, metadata = self.base_pipeline.retrieve(
            rephrased_query, 
            top_k=top_k * 3 if (use_object_filtering or use_text_filtering) else top_k,  # Get more results for multi-modal filtering
            use_reranking=use_reranking
        )
        
        # Step 3: Multi-modal filtering combining objects and OCR text
        if use_object_filtering or use_text_filtering:
            print("Step 3: Multi-modal filtering (Objects + OCR Text)...")
            
            # Initialize filtered results with all candidates
            filtered_results = []
            
            for i, meta in enumerate(metadata):
                embedding_score = scores[i]
                object_score = 0.0
                text_score = 0.0
                
                # Object-based filtering and scoring
                if use_object_filtering and suggested_objects:
                    object_filtered = self.filter_by_objects(
                        [meta], 
                        suggested_objects, 
                        confidence_threshold=object_confidence_threshold,
                        match_threshold=0.2  # Lower threshold for inclusion
                    )
                    if object_filtered:
                        object_score = object_filtered[0].get('object_match_score', 0.0)
                        meta.update(object_filtered[0])  # Update with object matching info
                
                # OCR text-based filtering and scoring
                if use_text_filtering:
                    text_filtered = self.filter_by_ocr_text(
                        [meta],
                        query,
                        text_similarity_threshold=text_similarity_threshold * 0.7  # Lower threshold for inclusion
                    )
                    if text_filtered:
                        text_score = text_filtered[0].get('text_match_score', 0.0)
                        meta.update(text_filtered[0])  # Update with text matching info
                
                # Multi-modal confidence scoring
                # Weighted combination: 60% embedding + 25% object + 15% text
                combined_score = (
                    0.60 * embedding_score + 
                    0.25 * object_score + 
                    0.15 * text_score
                )
                
                # Confidence boost for multi-modal matches
                confidence_boost = 0.0
                if object_score > 0 and text_score > 0:
                    confidence_boost = 0.05  # 5% boost for multi-modal match
                elif object_score > 0.5 or text_score > 0.5:
                    confidence_boost = 0.02  # 2% boost for strong single-modal match
                
                combined_score += confidence_boost
                
                # Include result if it meets any filtering criteria
                include_result = True
                if use_object_filtering and suggested_objects and use_text_filtering:
                    # Both filters enabled - require at least one to match above threshold
                    include_result = (object_score >= 0.3 or text_score >= text_similarity_threshold)
                elif use_object_filtering and suggested_objects:
                    # Only object filtering - require object match
                    include_result = (object_score >= 0.3)
                elif use_text_filtering:
                    # Only text filtering - require text match
                    include_result = (text_score >= text_similarity_threshold)
                
                if include_result:
                    filtered_results.append({
                        'metadata': meta,
                        'embedding_score': embedding_score,
                        'object_score': object_score,
                        'text_score': text_score,
                        'combined_score': combined_score,
                        'confidence_boost': confidence_boost,
                        'matched_objects': meta.get('matched_objects', []),
                        'matched_text_keywords': meta.get('matched_text_keywords', [])
                    })
            
            # Sort by combined confidence score
            filtered_results.sort(key=lambda x: x['combined_score'], reverse=True)
            
            # Extract final results
            final_metadata = [result['metadata'] for result in filtered_results[:top_k]]
            final_scores = [result['combined_score'] for result in filtered_results[:top_k]]
            
            # Add comprehensive scoring information to metadata
            for i, result in enumerate(filtered_results[:top_k]):
                final_metadata[i]['embedding_score'] = result['embedding_score']
                final_metadata[i]['object_score'] = result['object_score']
                final_metadata[i]['text_score'] = result['text_score']
                final_metadata[i]['combined_score'] = result['combined_score']
                final_metadata[i]['confidence_boost'] = result['confidence_boost']
                final_metadata[i]['multi_modal_match'] = (result['object_score'] > 0 and result['text_score'] > 0)
            
            # Logging
            print(f"Multi-modal filtering: {len(metadata)} → {len(filtered_results)} → {len(final_metadata)} results")
            if final_metadata:
                top_result = final_metadata[0]
                print(f"Top result scores - Combined: {top_result.get('combined_score', 0):.3f}, "
                      f"Object: {top_result.get('object_score', 0):.3f}, "
                      f"Text: {top_result.get('text_score', 0):.3f}")
                
                if top_result.get('matched_objects'):
                    print(f"Matched objects: {[obj['class'] for obj in top_result.get('matched_objects', [])[:3]]}")
                if top_result.get('matched_text_keywords'):
                    print(f"Matched text: {top_result.get('matched_text_keywords', [])[:5]}")
            
            return extraction_results, final_scores, final_metadata
        else:
            # No filtering - return original results
            return extraction_results, scores, metadata
            
    def save_index(self, 
                  index_path: str = Config.INDEX_PATH,
                  metadata_path: str = Config.METADATA_PATH) -> None:
        """Save FAISS index and metadata to disk"""
        self.base_pipeline.save_index(index_path, metadata_path)
        
    def load_index(self,
                  index_path: str = Config.INDEX_PATH,
                  metadata_path: str = Config.METADATA_PATH) -> None:
        """Load FAISS index and metadata from disk"""
        self.base_pipeline.load_index(index_path, metadata_path)
        
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the current index and pipeline"""
        base_stats = self.base_pipeline.get_stats()
        
        enhanced_stats = {
            "visual_extractor_model": "gemini-2.0-flash-exp",
            "has_object_detection": any(
                meta.get('has_object_detection', False) 
                for meta in getattr(self.base_pipeline.faiss_indexer, 'metadata', [])
            )
        }
        
        return {**base_stats, **enhanced_stats}