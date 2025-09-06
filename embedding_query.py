"""Text embedding query using Hugging Face CoCa models"""

import torch
import numpy as np
from transformers import AutoModel, AutoProcessor, CLIPModel, CLIPProcessor
from typing import Union, List
from config import Config


class EmbeddingQuery:
    def __init__(self, model_name: str = Config.EMBEDDING_MODEL):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model and processor
        try:
            # Try loading as CLIP model first
            self.model = CLIPModel.from_pretrained(model_name)
            self.processor = CLIPProcessor.from_pretrained(model_name)
            self.model_type = "clip"
        except:
            # Fallback to generic AutoModel
            self.model = AutoModel.from_pretrained(model_name)
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model_type = "auto"
            
        self.model.to(self.device)
        self.model.eval()
        
    def encode_text(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Encode text query into embedding vector
        
        Args:
            text: Single text string or list of text strings
            
        Returns:
            Normalized embedding vector(s) of shape (1, D) or (N, D)
        """
        if isinstance(text, str):
            text = [text]
            
        with torch.no_grad():
            if self.model_type == "clip":
                inputs = self.processor(text=text, return_tensors="pt", 
                                      padding=True, truncation=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                text_features = self.model.get_text_features(**inputs)
            else:
                inputs = self.processor(text=text, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                outputs = self.model(**inputs)
                text_features = outputs.last_hidden_state.mean(dim=1)
                
            # L2 normalize
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
        return text_features.cpu().numpy()
    
    def encode(self, query: str) -> np.ndarray:
        """
        Encode text query into embedding vector
        
        Args:
            query: Text string
            
        Returns:
            Normalized embedding vector of shape (1, D)
        """
        if not isinstance(query, str):
            raise ValueError("Query must be a text string")
        return self.encode_text(query)
            
    def get_embedding_dim(self) -> int:
        """Get the dimensionality of embeddings"""
        if self.model_type == "clip":
            return self.model.config.projection_dim
        else:
            return self.model.config.hidden_size