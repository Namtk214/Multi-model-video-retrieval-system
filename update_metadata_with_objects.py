"""Script to update metadata.json with object detection information from objects folder"""

import json
import os
from typing import List, Dict, Any
from tqdm import tqdm

def load_object_detection(object_file_path: str, confidence_threshold: float = 0.3) -> List[Dict[str, Any]]:
    """
    Load and parse object detection results from JSON file
    
    Args:
        object_file_path: Path to object detection JSON file
        confidence_threshold: Minimum confidence score to include objects
        
    Returns:
        List of detected objects with confidence > threshold
    """
    try:
        with open(object_file_path, 'r') as f:
            detection_data = json.load(f)
        
        # Parse detection data
        scores = [float(score) for score in detection_data.get('detection_scores', [])]
        class_names = detection_data.get('detection_class_entities', [])
        boxes = detection_data.get('detection_boxes', [])
        
        # Filter objects by confidence threshold
        filtered_objects = []
        for i, (score, class_name, box) in enumerate(zip(scores, class_names, boxes)):
            if score >= confidence_threshold:
                filtered_objects.append({
                    'class': class_name,
                    'confidence': score,
                    'bbox': [float(coord) for coord in box] if box else [],
                    'index': i
                })
        
        return filtered_objects
        
    except Exception as e:
        print(f"Error loading {object_file_path}: {e}")
        return []

def update_metadata_with_objects(metadata_path: str = "metadata.json", 
                                objects_base_dir: str = "objects",
                                confidence_threshold: float = 0.3,
                                backup_original: bool = True) -> None:
    """
    Update metadata.json with object detection information
    
    Args:
        metadata_path: Path to metadata.json file
        objects_base_dir: Base directory containing object detection results
        confidence_threshold: Minimum confidence threshold for objects
        backup_original: Whether to create a backup of original metadata
    """
    print(f"Updating metadata with object detection (threshold: {confidence_threshold})")
    
    # Create backup if requested
    if backup_original and os.path.exists(metadata_path):
        backup_path = f"{metadata_path}.backup"
        print(f"Creating backup: {backup_path}")
        with open(metadata_path, 'r') as src, open(backup_path, 'w') as dst:
            dst.write(src.read())
    
    # Load existing metadata
    print("Loading metadata...")
    with open(metadata_path, 'r') as f:
        metadata_list = json.load(f)
    
    print(f"Processing {len(metadata_list)} metadata entries...")
    
    updated_count = 0
    failed_count = 0
    
    for metadata in tqdm(metadata_list, desc="Processing frames"):
        video_id = metadata.get('video_id', '')
        frame_id = metadata.get('frame_id', '')
        frame_index = metadata.get('frame_index', 0)
        
        if not video_id or not frame_id:
            continue
        
        # Construct object detection file path
        video_dir = f"{video_id}_{frame_id}"
        frame_filename = f"{frame_index + 1:03d}.json"  # Frame numbers start from 1, zero-padded
        object_file_path = os.path.join(objects_base_dir, video_dir, frame_filename)
        
        if os.path.exists(object_file_path):
            # Load object detection results
            detected_objects = load_object_detection(object_file_path, confidence_threshold)
            
            # Update metadata
            metadata['detected_objects'] = detected_objects
            metadata['object_detection_file'] = object_file_path
            metadata['has_object_detection'] = True
            metadata['num_detected_objects'] = len(detected_objects)
            
            # Extract unique object classes for quick filtering
            metadata['object_classes'] = list(set(obj['class'] for obj in detected_objects))
            
            updated_count += 1
        else:
            # No object detection file found
            metadata['detected_objects'] = []
            metadata['object_detection_file'] = None
            metadata['has_object_detection'] = False
            metadata['num_detected_objects'] = 0
            metadata['object_classes'] = []
            failed_count += 1
    
    # Save updated metadata
    print("Saving updated metadata...")
    with open(metadata_path, 'w') as f:
        json.dump(metadata_list, f, indent=2)
    
    print(f"\n✅ Update Complete!")
    print(f"  - Total frames: {len(metadata_list)}")
    print(f"  - Updated with objects: {updated_count}")
    print(f"  - No object detection: {failed_count}")
    print(f"  - Confidence threshold: {confidence_threshold}")

def analyze_object_distribution(metadata_path: str = "metadata.json") -> Dict[str, int]:
    """
    Analyze the distribution of detected objects across all frames
    
    Args:
        metadata_path: Path to updated metadata.json
        
    Returns:
        Dictionary of object class counts
    """
    with open(metadata_path, 'r') as f:
        metadata_list = json.load(f)
    
    object_counts = {}
    
    for metadata in metadata_list:
        if metadata.get('has_object_detection', False):
            for obj_class in metadata.get('object_classes', []):
                object_counts[obj_class] = object_counts.get(obj_class, 0) + 1
    
    return object_counts

def main():
    """Main function to update metadata and analyze results"""
    
    # Update metadata with object detection info (confidence > 0.3)
    update_metadata_with_objects(
        metadata_path="metadata.json",
        objects_base_dir="objects",
        confidence_threshold=0.3,
        backup_original=True
    )
    
    # Analyze object distribution
    print("\n📊 Object Distribution Analysis:")
    object_counts = analyze_object_distribution()
    
    # Show top 20 most common objects
    sorted_objects = sorted(object_counts.items(), key=lambda x: x[1], reverse=True)
    print("\nTop 20 Most Common Objects:")
    for i, (obj_class, count) in enumerate(sorted_objects[:20], 1):
        print(f"  {i:2d}. {obj_class:<20} : {count:>5} frames")
    
    print(f"\nTotal unique object classes: {len(object_counts)}")

if __name__ == "__main__":
    main()