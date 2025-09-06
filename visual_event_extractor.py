"""Visual Event Extractor for processing queries into structured visual elements"""

import json
import re
from typing import Dict, List, Any, Optional
import google.generativeai as genai
from config import Config


class VisualEventExtractor:
    """Extracts visual elements, actions, and objects from natural language queries using LLM"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Visual Event Extractor with Gemini 2.5 Flash
        
        Args:
            api_key: Google AI API key. If None, expects GOOGLE_API_KEY env variable
        """
        if api_key:
            genai.configure(api_key=api_key)
        
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        self.extraction_prompt = """
You are an expert at analyzing visual queries and extracting structured information for video retrieval.

Given a natural language query, extract the following information in JSON format:

1. Visual Elements: Key visual components, objects, people, locations
2. Actions: Verbs and activities being performed
3. Suggested Objects: Specific object classes that should be detected in frames
4. Rephrased Query: A more searchable version of the original query optimized for embedding similarity

Original Query: "{query}"

Please respond with a JSON object containing:
- "visual_elements": List of visual components
- "actions": List of actions/verbs
- "suggested_objects": List of specific object classes for detection
- "rephrased_query": A more searchable version of the query

Example format:
{{
    "visual_elements": ["person", "kitchen", "pasta", "utensils"],
    "actions": ["cooking", "stirring", "boiling"],
    "suggested_objects": ["person", "bowl", "spoon", "stove", "pot"],
    "rephrased_query": "person cooking food in kitchen with utensils"
}}
"""

    def extract_visual_events(self, query: str) -> Dict[str, Any]:
        """
        Extract visual events from a natural language query
        
        Args:
            query: Natural language query string
            
        Returns:
            Dictionary containing extracted visual elements, actions, objects, and rephrased query
        """
        try:
            # Format the prompt with the query
            formatted_prompt = self.extraction_prompt.format(query=query)
            
            # Generate response using Gemini
            response = self.model.generate_content(formatted_prompt)
            
            # Extract JSON from response
            response_text = response.text
            
            # Try to find JSON in the response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                extracted_data = json.loads(json_str)
            else:
                # Fallback: try to parse the entire response as JSON
                extracted_data = json.loads(response_text)
            
            # Validate required fields
            required_fields = ["visual_elements", "actions", "suggested_objects", "rephrased_query"]
            for field in required_fields:
                if field not in extracted_data:
                    extracted_data[field] = []
            
            # Ensure rephrased_query is a string
            if not isinstance(extracted_data["rephrased_query"], str):
                extracted_data["rephrased_query"] = query
                
            return extracted_data
            
        except Exception as e:
            print(f"Error extracting visual events: {e}")
            # Return default structure with original query
            return {
                "visual_elements": [],
                "actions": [],
                "suggested_objects": [],
                "rephrased_query": query
            }
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a query through the complete visual event extraction pipeline
        
        Args:
            query: Original natural language query
            
        Returns:
            Dictionary with original query, extracted elements, and processing metadata
        """
        extraction_result = self.extract_visual_events(query)
        
        return {
            "original_query": query,
            "visual_elements": extraction_result["visual_elements"],
            "actions": extraction_result["actions"],
            "suggested_objects": extraction_result["suggested_objects"],
            "rephrased_query": extraction_result["rephrased_query"],
            "extraction_metadata": {
                "model_used": "gemini-2.0-flash-exp",
                "extraction_successful": len(extraction_result["rephrased_query"]) > 0
            }
        }