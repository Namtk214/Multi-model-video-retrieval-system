"""Script to update metadata.json with OCR text information from video frames"""

import json
import os
from typing import List, Dict, Any
from tqdm import tqdm
from pathlib import Path

from ocr_processor import OCRProcessor, load_mistral_api_key


def update_metadata_with_ocr(metadata_path: str = "metadata.json", 
                            ocr_cache_path: str = "ocr_cache.json",
                            batch_size: int = 10,
                            max_frames: int = None,
                            backup_original: bool = True) -> None:
    """
    Update metadata.json with OCR text information from video frames
    
    Args:
        metadata_path: Path to metadata.json file
        ocr_cache_path: Path to cache OCR results
        batch_size: Number of images to process in each batch
        max_frames: Maximum number of frames to process (for testing)
        backup_original: Whether to create a backup of original metadata
    """
    print("=== OCR Text Extraction Pipeline ===")
    
    # Load API key
    api_key = load_mistral_api_key()
    if not api_key:
        print("❌ Mistral API key not found. Please add MISTRAL_API_KEY to key.env file")
        return
    
    # Initialize OCR processor
    ocr_processor = OCRProcessor(api_key=api_key)
    if not ocr_processor.is_available():
        print("❌ OCR processor not available")
        return
    
    print("✅ OCR processor initialized")
    
    # Create backup if requested
    if backup_original and os.path.exists(metadata_path):
        backup_path = f"{metadata_path}.ocr_backup"
        print(f"Creating backup: {backup_path}")
        with open(metadata_path, 'r') as src, open(backup_path, 'w') as dst:
            dst.write(src.read())
    
    # Load existing OCR cache
    ocr_cache = {}
    if os.path.exists(ocr_cache_path):
        try:
            with open(ocr_cache_path, 'r') as f:
                ocr_cache = json.load(f)
            print(f"✅ Loaded OCR cache with {len(ocr_cache)} entries")
        except Exception as e:
            print(f"⚠ Error loading OCR cache: {e}")
            ocr_cache = {}
    
    # Load metadata
    print("Loading metadata...")
    with open(metadata_path, 'r') as f:
        metadata_list = json.load(f)
    
    # Limit for testing if specified
    if max_frames:
        metadata_list = metadata_list[:max_frames]
        print(f"🔬 Testing mode: processing first {max_frames} frames")
    
    print(f"Processing {len(metadata_list)} metadata entries...")
    
    # Filter frames that need OCR processing
    frames_to_process = []
    for metadata in metadata_list:
        image_path = metadata.get('image_path', '')
        if image_path and os.path.exists(image_path) and image_path not in ocr_cache:
            frames_to_process.append((image_path, metadata))
    
    print(f"Found {len(frames_to_process)} new frames to process (cached: {len(ocr_cache)})")
    
    if frames_to_process:
        # Extract image paths for batch processing
        image_paths = [path for path, _ in frames_to_process]
        
        print("🔍 Starting OCR text extraction...")
        
        # Process in batches
        ocr_results = ocr_processor.process_batch_images(
            image_paths, 
            batch_size=batch_size,
            delay_between_batches=2.0  # 2 second delay between batches
        )
        
        # Update cache with new results
        ocr_cache.update(ocr_results)
        
        # Save cache
        print("💾 Saving OCR cache...")
        with open(ocr_cache_path, 'w') as f:
            json.dump(ocr_cache, f, indent=2)
    
    # Update metadata with OCR information
    print("📝 Updating metadata with OCR results...")
    
    updated_count = 0
    text_found_count = 0
    
    for metadata in tqdm(metadata_list, desc="Processing metadata"):
        image_path = metadata.get('image_path', '')
        
        if image_path in ocr_cache:
            ocr_result = ocr_cache[image_path]
            
            # Add OCR information to metadata
            metadata['ocr_text'] = ocr_result.get('text', '')
            metadata['ocr_success'] = ocr_result.get('success', False)
            metadata['ocr_confidence'] = ocr_result.get('confidence', 0.0)
            metadata['ocr_word_count'] = ocr_result.get('word_count', 0)
            metadata['ocr_char_count'] = ocr_result.get('char_count', 0)
            metadata['ocr_processing_time'] = ocr_result.get('processing_time', 0.0)
            metadata['ocr_model'] = ocr_result.get('model_used', 'mistral-ocr-latest')
            metadata['has_ocr_text'] = len(ocr_result.get('text', '')) > 0
            
            # Extract keywords from OCR text
            if ocr_result.get('text'):
                keywords = ocr_processor.extract_keywords_from_text(ocr_result['text'])
                metadata['ocr_keywords'] = keywords
                metadata['ocr_keyword_count'] = len(keywords)
                text_found_count += 1
            else:
                metadata['ocr_keywords'] = []
                metadata['ocr_keyword_count'] = 0
            
            updated_count += 1
        else:
            # No OCR processing for this frame
            metadata['ocr_text'] = ''
            metadata['ocr_success'] = False
            metadata['ocr_confidence'] = 0.0
            metadata['ocr_word_count'] = 0
            metadata['ocr_char_count'] = 0
            metadata['ocr_processing_time'] = 0.0
            metadata['ocr_model'] = None
            metadata['has_ocr_text'] = False
            metadata['ocr_keywords'] = []
            metadata['ocr_keyword_count'] = 0
    
    # Save updated metadata
    print("💾 Saving updated metadata...")
    with open(metadata_path, 'w') as f:
        json.dump(metadata_list, f, indent=2)
    
    print(f"\n✅ OCR Processing Complete!")
    print(f"  - Total frames: {len(metadata_list)}")
    print(f"  - Updated with OCR: {updated_count}")
    print(f"  - Frames with text: {text_found_count}")
    print(f"  - OCR cache size: {len(ocr_cache)}")


def analyze_ocr_results(metadata_path: str = "metadata.json") -> Dict[str, Any]:
    """
    Analyze OCR results in the metadata
    
    Args:
        metadata_path: Path to updated metadata.json
        
    Returns:
        Dictionary with OCR analysis results
    """
    with open(metadata_path, 'r') as f:
        metadata_list = json.load(f)
    
    stats = {
        'total_frames': len(metadata_list),
        'frames_with_text': 0,
        'total_words': 0,
        'total_characters': 0,
        'average_words_per_frame': 0,
        'most_common_keywords': {},
        'processing_time': 0,
        'success_rate': 0
    }
    
    keyword_counts = {}
    successful_ocr = 0
    total_processing_time = 0
    
    for metadata in metadata_list:
        if metadata.get('has_ocr_text', False):
            stats['frames_with_text'] += 1
            stats['total_words'] += metadata.get('ocr_word_count', 0)
            stats['total_characters'] += metadata.get('ocr_char_count', 0)
        
        if metadata.get('ocr_success', False):
            successful_ocr += 1
            total_processing_time += metadata.get('ocr_processing_time', 0)
        
        # Count keywords
        for keyword in metadata.get('ocr_keywords', []):
            keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
    
    # Calculate averages and rates
    stats['success_rate'] = successful_ocr / len(metadata_list) if metadata_list else 0
    stats['average_words_per_frame'] = stats['total_words'] / stats['frames_with_text'] if stats['frames_with_text'] else 0
    stats['processing_time'] = total_processing_time
    
    # Get most common keywords
    sorted_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)
    stats['most_common_keywords'] = dict(sorted_keywords[:20])
    
    return stats


def main():
    """Main function to run OCR processing"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Update metadata with OCR text extraction')
    parser.add_argument('--metadata', default='metadata.json', help='Path to metadata.json file')
    parser.add_argument('--cache', default='ocr_cache.json', help='Path to OCR cache file')
    parser.add_argument('--batch-size', type=int, default=10, help='Batch size for processing')
    parser.add_argument('--max-frames', type=int, help='Maximum frames to process (for testing)')
    parser.add_argument('--analyze-only', action='store_true', help='Only analyze existing OCR results')
    
    args = parser.parse_args()
    
    if args.analyze_only:
        print("📊 Analyzing existing OCR results...")
        stats = analyze_ocr_results(args.metadata)
        
        print(f"\n=== OCR Analysis Results ===")
        print(f"Total frames: {stats['total_frames']}")
        print(f"Frames with text: {stats['frames_with_text']}")
        print(f"Success rate: {stats['success_rate']:.1%}")
        print(f"Total words extracted: {stats['total_words']}")
        print(f"Average words per frame: {stats['average_words_per_frame']:.1f}")
        print(f"Total processing time: {stats['processing_time']:.1f}s")
        
        print(f"\nTop 10 Keywords:")
        for i, (keyword, count) in enumerate(list(stats['most_common_keywords'].items())[:10], 1):
            print(f"  {i:2d}. {keyword:<15} : {count:>4} frames")
    
    else:
        # Run OCR processing
        update_metadata_with_ocr(
            metadata_path=args.metadata,
            ocr_cache_path=args.cache,
            batch_size=args.batch_size,
            max_frames=args.max_frames
        )
        
        # Show analysis
        print("\n📊 Post-processing Analysis:")
        stats = analyze_ocr_results(args.metadata)
        print(f"Frames with text: {stats['frames_with_text']}/{stats['total_frames']} ({stats['frames_with_text']/stats['total_frames']*100:.1f}%)")
        print(f"Total words extracted: {stats['total_words']}")
        print(f"Processing time: {stats['processing_time']:.1f}s")


if __name__ == "__main__":
    main()