"""
Embeddings Module
Handles generation of embeddings for text chunks
"""
from typing import List
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: sentence-transformers import failed: {e}")
    print("Installing compatible version...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "sentence-transformers==2.3.1", "huggingface-hub==0.20.3"])
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True

from config.config import Config


class EmbeddingGenerator:
    """Generate embeddings for text using sentence transformers"""
    
    def __init__(self, model_name: str = None):
        """
        Initialize the embedding generator
        
        Args:
            model_name: Name of the sentence transformer model
        """
        self.model_name = model_name or Config.EMBEDDING_MODEL
        print(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts: List of text strings
            
        Returns:
            numpy array of embeddings
        """
        if not texts:
            return np.array([])
        
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        return embeddings
    
    def generate_single_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text
        
        Args:
            text: Text string
            
        Returns:
            numpy array of embedding
        """
        embedding = self.model.encode(
            text,
            convert_to_numpy=True
        )
        
        return embedding
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings"""
        return self.embedding_dim
