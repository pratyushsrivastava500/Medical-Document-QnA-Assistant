"""
Configuration Management Module
Loads and manages environment variables and application settings
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Application configuration"""
    
    # API Keys
    EURIAI_API_KEY = os.getenv("EURIAI_API_KEY", "euri-3412eeb7b59f9b5444366350489a866dc2b01217fbe56b420c1612e1d845cad6")
    
    # Model Configuration
    EURIAI_MODEL = os.getenv("EURIAI_MODEL", "gpt-4.1-nano")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    
    # Document Processing
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
    TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "5"))
    
    # Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    UPLOAD_DIR = os.path.join(DATA_DIR, "uploaded")
    VECTOR_DB_DIR = os.path.join(DATA_DIR, "vector_db")
    GDRIVE_CACHE_DIR = os.path.join(DATA_DIR, "gdrive_cache")
    
    @classmethod
    def validate(cls):
        """Validate required configuration"""
        if not cls.EURIAI_API_KEY:
            raise ValueError("EURIAI_API_KEY is not set in environment variables")
        
        # Create necessary directories
        for directory in [cls.DATA_DIR, cls.UPLOAD_DIR, cls.VECTOR_DB_DIR, cls.GDRIVE_CACHE_DIR]:
            os.makedirs(directory, exist_ok=True)


# Validate configuration on import
Config.validate()
