"""OCR Processing Pipeline using Mistral API for video frame text extraction"""

import os
import base64
import re
import json
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import time
from tqdm import tqdm

try:
    from mistralai import Mistral
    MISTRAL_AVAILABLE = True
except ImportError:
    MISTRAL_AVAILABLE = False
    print("Warning: mistralai package not installed. OCR functionality will be disabled.")
    print("Install with: pip install mistralai")


class OCRProcessor:
    """OCR processor using Mistral API for text extraction from images"""
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "mistral-ocr-latest"):
        """
        Initialize OCR processor with Mistral API
        
        Args:
            api_key: Mistral API key. If None, expects MISTRAL_API_KEY env variable
            model_name: Mistral OCR model to use
        """
        self.model_name = model_name
        self.api_key = api_key or os.getenv('MISTRAL_API_KEY')
        
        if not MISTRAL_AVAILABLE:
            print("❌ Mistral AI package not available")
            self.client = None
            return
            
        if not self.api_key:
            print("❌ Mistral API key not found. Set MISTRAL_API_KEY environment variable or add to key.env")
            self.client = None
            return
            
        try:
            self.client = Mistral(api_key=self.api_key)
            print("✅ Mistral OCR client initialized successfully")
        except Exception as e:
            print(f"❌ Error initializing Mistral client: {e}")
            self.client = None
    
    def is_available(self) -> bool:
        """Check if OCR processing is available"""
        return MISTRAL_AVAILABLE and self.client is not None
    
    def encode_image(self, image_path: str) -> Optional[str]:
        """
        Encode image to base64 string
        
        Args:
            image_path: Path to image file
            
        Returns:
            Base64 encoded image string or None if error
        """
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        except FileNotFoundError:
            print(f"Error: {image_path} not found.")
            return None
        except Exception as e:
            print(f"Error encoding image {image_path}: {e}")
            return None
    
    def extract_text_from_image(self, image_path: str, include_image_data: bool = False) -> Dict[str, Any]:
        """
        Extract text from image using Mistral OCR
        
        Args:
            image_path: Path to image file
            include_image_data: Whether to include base64 image data in response
            
        Returns:
            Dictionary with extracted text and metadata
        """
        if not self.is_available():
            return {
                'text': '',
                'success': False,
                'error': 'OCR service not available',
                'confidence': 0.0,
                'processing_time': 0.0
            }
        
        try:
            start_time = time.time()
            
            # Encode image to base64
            base64_image = self.encode_image(image_path)
            if not base64_image:
                return {
                    'text': '',
                    'success': False,
                    'error': 'Failed to encode image',
                    'confidence': 0.0,
                    'processing_time': 0.0
                }
            
            # Determine image format from file extension
            image_format = Path(image_path).suffix.lower().lstrip('.')
            if image_format not in ['jpg', 'jpeg', 'png', 'gif', 'bmp']:
                image_format = 'jpeg'
            
            # Call Mistral OCR API
            ocr_response = self.client.ocr.process(
                model=self.model_name,
                document={
                    "type": "image_url",
                    "image_url": f"data:image/{image_format};base64,{base64_image}"
                },
                include_image_base64=include_image_data
            )
            
            processing_time = time.time() - start_time
            
            # Extract text from response
            extracted_text = ""
            confidence = 1.0  # Mistral doesn't provide confidence scores
            
            if hasattr(ocr_response, 'text'):
                extracted_text = ocr_response.text
            elif hasattr(ocr_response, 'content'):
                extracted_text = str(ocr_response.content)
            elif isinstance(ocr_response, dict):
                extracted_text = ocr_response.get('text', str(ocr_response))
            else:
                extracted_text = str(ocr_response)
            
            # Clean and process text
            cleaned_text = self._clean_extracted_text(extracted_text)
            
            return {
                'text': cleaned_text,
                'raw_text': extracted_text,
                'success': True,
                'error': None,
                'confidence': confidence,
                'processing_time': processing_time,
                'model_used': self.model_name,
                'word_count': len(cleaned_text.split()) if cleaned_text else 0,
                'char_count': len(cleaned_text) if cleaned_text else 0
            }
            
        except Exception as e:
            processing_time = time.time() - start_time if 'start_time' in locals() else 0.0
            return {
                'text': '',
                'success': False,
                'error': str(e),
                'confidence': 0.0,
                'processing_time': processing_time
            }
    
    def _clean_extracted_text(self, text: str) -> str:
        """
        Clean and normalize extracted text
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text string
        """
        if not text:
            return ""
        
        # Remove extra whitespace and normalize line breaks
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove common OCR artifacts
        text = re.sub(r'[^\w\s\-.,!?;:()\[\]{}"\'/]', ' ', text)
        
        # Normalize Vietnamese characters if present
        text = self._normalize_vietnamese_text(text)
        
        # Remove excessive punctuation
        text = re.sub(r'([.!?]){2,}', r'\1', text)
        
        return text.strip()
    
    def _normalize_vietnamese_text(self, text: str) -> str:
        """
        Normalize Vietnamese text for better matching
        
        Args:
            text: Input text with Vietnamese characters
            
        Returns:
            Normalized text
        """
        # Common Vietnamese character normalizations
        vietnamese_mappings = {
            'à|á|ạ|ả|ã|â|ầ|ấ|ậ|ẩ|ẫ|ă|ằ|ắ|ặ|ẳ|ẵ': 'a',
            'è|é|ẹ|ẻ|ẽ|ê|ề|ế|ệ|ể|ễ': 'e',
            'ì|í|ị|ỉ|ĩ': 'i',
            'ò|ó|ọ|ỏ|õ|ô|ồ|ố|ộ|ổ|ỗ|ơ|ờ|ớ|ợ|ở|ỡ': 'o',
            'ù|ú|ụ|ủ|ũ|ư|ừ|ứ|ự|ử|ữ': 'u',
            'ỳ|ý|ỵ|ỷ|ỹ': 'y',
            'đ': 'd'
        }
        
        # Apply normalizations (optional - keep original for exact matching)
        normalized_text = text
        for pattern, replacement in vietnamese_mappings.items():
            normalized_text = re.sub(f'[{pattern}]', replacement, normalized_text, flags=re.IGNORECASE)
        
        return text  # Return original text to preserve Vietnamese characters
    
    def extract_keywords_from_text(self, text: str, min_length: int = 3) -> List[str]:
        """
        Extract meaningful keywords from OCR text
        
        Args:
            text: Extracted text
            min_length: Minimum keyword length
            
        Returns:
            List of keywords
        """
        if not text:
            return []
        
        # Split text into words
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Filter words by length and remove common stop words
        stop_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'từ', 'và', 'hoặc', 'nhưng', 'trong', 'trên', 'tại', 'để', 'của', 'với', 'bởi',
            'một', 'này', 'đó', 'những', 'các', 'có', 'là', 'được', 'sẽ', 'đã'
        }
        
        keywords = [word for word in words 
                   if len(word) >= min_length and word not in stop_words]
        
        # Remove duplicates while preserving order
        unique_keywords = []
        seen = set()
        for keyword in keywords:
            if keyword not in seen:
                unique_keywords.append(keyword)
                seen.add(keyword)
        
        return unique_keywords[:20]  # Return top 20 keywords
    
    def process_batch_images(self, image_paths: List[str], 
                           batch_size: int = 5,
                           delay_between_batches: float = 1.0) -> Dict[str, Dict[str, Any]]:
        """
        Process multiple images in batches to respect API rate limits
        
        Args:
            image_paths: List of image paths to process
            batch_size: Number of images to process in each batch
            delay_between_batches: Delay between batches in seconds
            
        Returns:
            Dictionary mapping image paths to OCR results
        """
        if not self.is_available():
            return {path: {
                'text': '', 'success': False, 'error': 'OCR service not available'
            } for path in image_paths}
        
        results = {}
        total_batches = (len(image_paths) + batch_size - 1) // batch_size
        
        for batch_idx in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[batch_idx:batch_idx + batch_size]
            batch_num = (batch_idx // batch_size) + 1
            
            print(f"Processing batch {batch_num}/{total_batches} ({len(batch_paths)} images)...")
            
            for image_path in tqdm(batch_paths, desc=f"Batch {batch_num}"):
                try:
                    result = self.extract_text_from_image(image_path)
                    results[image_path] = result
                    
                    # Small delay between individual requests
                    time.sleep(0.1)
                    
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")
                    results[image_path] = {
                        'text': '',
                        'success': False,
                        'error': str(e),
                        'confidence': 0.0,
                        'processing_time': 0.0
                    }
            
            # Delay between batches to respect rate limits
            if batch_num < total_batches:
                print(f"Waiting {delay_between_batches}s before next batch...")
                time.sleep(delay_between_batches)
        
        return results


def load_mistral_api_key() -> Optional[str]:
    """Load Mistral API key from key.env file or environment"""
    try:
        # Try loading from key.env file
        key_file = "key.env"
        if os.path.exists(key_file):
            with open(key_file, 'r') as f:
                content = f.read()
            
            # Look for MISTRAL_API_KEY pattern
            pattern = r"MISTRAL_API_KEY\s*=\s*['\"]([^'\"]+)['\"]"
            match = re.search(pattern, content)
            if match:
                return match.group(1)
    except Exception:
        pass
    
    # Fallback to environment variable
    return os.getenv('MISTRAL_API_KEY')


# Example usage and testing
if __name__ == "__main__":
    # Test OCR processor
    api_key = load_mistral_api_key()
    
    if not api_key:
        print("Please set MISTRAL_API_KEY in key.env file or as environment variable")
        exit(1)
    
    ocr = OCRProcessor(api_key=api_key)
    
    if not ocr.is_available():
        print("OCR processor not available")
        exit(1)
    
    # Test with a sample image (if available)
    test_image = "keyframes/L21_V001/001.jpg"  # Adjust path as needed
    
    if os.path.exists(test_image):
        print(f"Testing OCR with: {test_image}")
        result = ocr.extract_text_from_image(test_image)
        
        print(f"Success: {result['success']}")
        print(f"Text: {result['text']}")
        print(f"Words: {result.get('word_count', 0)}")
        print(f"Processing time: {result['processing_time']:.2f}s")
        
        if result['text']:
            keywords = ocr.extract_keywords_from_text(result['text'])
            print(f"Keywords: {keywords}")
    else:
        print(f"Test image not found: {test_image}")
        print("OCR processor initialized successfully - ready for use")