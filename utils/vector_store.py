"""
Vector Store Module
Manages vector database for semantic search using FAISS
"""
import os
import pickle
from typing import List, Dict, Any, Tuple
import numpy as np
import faiss
from config.config import Config


class VectorStore:
    """Vector database for storing and retrieving document embeddings using FAISS"""
    
    def __init__(self, collection_name: str = "medical_documents"):
        """
        Initialize the vector store
        
        Args:
            collection_name: Name of the collection
        """
        self.collection_name = collection_name
        self.persist_directory = Config.VECTOR_DB_DIR
        self.index_file = os.path.join(self.persist_directory, f"{collection_name}_faiss.index")
        self.metadata_file = os.path.join(self.persist_directory, f"{collection_name}_metadata.pkl")
        
        # Initialize FAISS index and metadata storage
        self.index = None
        self.documents = []  # Store document content
        self.metadatas = []  # Store document metadata
        self.dimension = 384  # Default dimension for all-MiniLM-L6-v2
        
        # Load existing index if available
        self._load_index()
        
        print(f"Vector store initialized with {len(self.documents)} documents")
    
    def _load_index(self):
        """Load existing FAISS index and metadata from disk"""
        if os.path.exists(self.index_file) and os.path.exists(self.metadata_file):
            try:
                # Load FAISS index
                self.index = faiss.read_index(self.index_file)
                self.dimension = self.index.d
                
                # Load metadata
                with open(self.metadata_file, 'rb') as f:
                    data = pickle.load(f)
                    self.documents = data.get('documents', [])
                    self.metadatas = data.get('metadatas', [])
                
                print(f"Loaded existing index with {len(self.documents)} documents")
            except Exception as e:
                print(f"Error loading index: {e}")
                self._create_new_index()
        else:
            self._create_new_index()
    
    def _create_new_index(self):
        """Create a new FAISS index"""
        # Use L2 distance (can also use Inner Product for cosine similarity)
        self.index = faiss.IndexFlatL2(self.dimension)
        self.documents = []
        self.metadatas = []
        print("Created new FAISS index")
    
    def _save_index(self):
        """Save FAISS index and metadata to disk"""
        try:
            # Save FAISS index
            faiss.write_index(self.index, self.index_file)
            
            # Save metadata
            with open(self.metadata_file, 'wb') as f:
                pickle.dump({
                    'documents': self.documents,
                    'metadatas': self.metadatas
                }, f)
            
            print(f"Saved index with {len(self.documents)} documents")
        except Exception as e:
            print(f"Error saving index: {e}")
    
    def add_documents(self, documents: List[Dict[str, Any]], 
                     embeddings: np.ndarray) -> None:
        """
        Add documents and their embeddings to the vector store
        
        Args:
            documents: List of document dictionaries with content and metadata
            embeddings: numpy array of embeddings
        """
        if len(documents) == 0:
            return
        
        # Update dimension if this is first addition
        if self.index.ntotal == 0 and embeddings.shape[1] != self.dimension:
            self.dimension = embeddings.shape[1]
            self._create_new_index()
        
        # Normalize embeddings for cosine similarity (optional)
        faiss.normalize_L2(embeddings)
        
        # Add to FAISS index
        self.index.add(embeddings.astype('float32'))
        
        # Store documents and metadata
        for doc in documents:
            self.documents.append(doc['content'])
            self.metadatas.append(doc['metadata'])
        
        # Save to disk
        self._save_index()
        
        print(f"Added {len(documents)} documents to vector store")
    
    def search(self, query_embedding: np.ndarray, 
              top_k: int = None) -> List[Dict[str, Any]]:
        """
        Search for similar documents
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            
        Returns:
            List of matching documents with metadata and scores
        """
        if top_k is None:
            top_k = Config.TOP_K_RESULTS
        
        if self.index.ntotal == 0:
            return []
        
        # Ensure query_embedding is the right shape
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Normalize for cosine similarity
        faiss.normalize_L2(query_embedding)
        
        # Search
        top_k = min(top_k, self.index.ntotal)  # Can't retrieve more than we have
        distances, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        # Format results
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.documents):  # Valid index
                # Convert L2 distance to similarity score (higher is better)
                similarity = 1 / (1 + distance)
                
                doc = {
                    'content': self.documents[idx],
                    'metadata': self.metadatas[idx],
                    'score': float(similarity)
                }
                results.append(doc)
        
        return results
    
    def delete_by_source(self, source_name: str) -> None:
        """
        Delete all documents from a specific source
        Note: FAISS doesn't support deletion, so we rebuild the index
        
        Args:
            source_name: Name of the source document
        """
        # Find indices to keep
        indices_to_keep = []
        new_documents = []
        new_metadatas = []
        
        for i, metadata in enumerate(self.metadatas):
            if metadata.get('source') != source_name:
                indices_to_keep.append(i)
                new_documents.append(self.documents[i])
                new_metadatas.append(metadata)
        
        if len(indices_to_keep) < len(self.documents):
            # Rebuild index without deleted documents
            self.documents = new_documents
            self.metadatas = new_metadatas
            
            # Note: This is a simplified version. In production, you'd need to 
            # regenerate embeddings or store them separately
            print(f"Marked documents from {source_name} for deletion")
            print("Note: Restart the app to rebuild the index without these documents")
    
    def clear_all(self) -> None:
        """Clear all documents from the vector store"""
        self._create_new_index()
        self._save_index()
        print("Vector store cleared")
    
    def get_all_sources(self) -> List[str]:
        """Get list of all unique source documents"""
        try:
            sources = set([meta.get('source', 'Unknown') for meta in self.metadatas])
            return sorted(list(sources))
        except Exception as e:
            print(f"Error getting sources: {e}")
            return []
    
    def count(self) -> int:
        """Get total number of documents in the vector store"""
        return len(self.documents)
