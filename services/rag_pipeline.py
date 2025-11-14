"""
RAG Pipeline Module
Implements Retrieval-Augmented Generation for grounded Q&A
"""
import os
from typing import List, Dict, Any, Tuple
from services.llm_client import LLMClient
from services.conversation_memory import ConversationMemory
from utils.embeddings import EmbeddingGenerator
from utils.vector_store import VectorStore


class RAGPipeline:
    """Retrieval-Augmented Generation pipeline for document Q&A"""
    
    def __init__(self, llm_client: LLMClient, 
                 embedding_generator: EmbeddingGenerator,
                 vector_store: VectorStore,
                 conversation_memory: ConversationMemory):
        """
        Initialize RAG pipeline
        
        Args:
            llm_client: LLM client instance
            embedding_generator: Embedding generator instance
            vector_store: Vector store instance
            conversation_memory: Conversation memory instance
        """
        self.llm_client = llm_client
        self.embedding_generator = embedding_generator
        self.vector_store = vector_store
        self.conversation_memory = conversation_memory
    
    def query(self, user_question: str, 
             top_k: int = 5,
             use_conversation_history: bool = True,
             selected_sources: List[str] = None) -> Dict[str, Any]:
        """
        Process a user query and generate a grounded response
        
        Args:
            user_question: User's question
            top_k: Number of relevant documents to retrieve
            use_conversation_history: Whether to include conversation context
            selected_sources: Optional list of source files to filter results
            
        Returns:
            Dictionary with answer, citations, and metadata
        """
        # 1. Retrieve relevant documents
        relevant_docs = self._retrieve_documents(user_question, top_k, selected_sources)
        
        if not relevant_docs:
            return {
                "answer": "This information is not available in the provided documents. Please upload documents or connect to Google Drive first.",
                "citations": [],
                "sources": [],
                "confidence": "none"
            }
        
        # 2. Check if retrieved documents have sufficient relevance
        # If all documents have very low relevance scores, the information might not be available
        avg_score = sum(doc['score'] for doc in relevant_docs) / len(relevant_docs)
        if avg_score < 0.3:  # Very low relevance threshold
            return {
                "answer": "This information is not available in the provided documents. The documents do not contain relevant information to answer this question.",
                "citations": [],
                "sources": [],
                "confidence": "low"
            }
        
        # 3. Build context from retrieved documents
        context = self._build_context(relevant_docs)
        
        # 4. Generate response with citations
        response = self._generate_grounded_response(
            user_question, 
            context, 
            relevant_docs,
            use_conversation_history
        )
        
        # 5. Update conversation memory
        self.conversation_memory.add_user_message(user_question)
        self.conversation_memory.add_assistant_message(
            response["answer"],
            response["citations"]
        )
        
        return response
    
    def _retrieve_documents(self, query: str, top_k: int, 
                          selected_sources: List[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents from vector store
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            selected_sources: Optional list of sources to filter results
            
        Returns:
            List of relevant documents
        """
        # Generate query embedding
        query_embedding = self.embedding_generator.generate_single_embedding(query)
        
        # Search vector store
        relevant_docs = self.vector_store.search(query_embedding, top_k)
        
        # Filter by selected sources if specified
        if selected_sources:
            relevant_docs = [doc for doc in relevant_docs 
                           if doc.get('metadata', {}).get('source') in selected_sources]
        
        return relevant_docs
    
    def _build_context(self, documents: List[Dict[str, Any]]) -> str:
        """
        Build context string from retrieved documents
        
        Args:
            documents: List of retrieved documents
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for i, doc in enumerate(documents, 1):
            source = doc['metadata'].get('source', 'Unknown')
            content = doc['content']
            chunk_id = doc['metadata'].get('chunk_id', 0)
            
            context_part = f"[Document {i}: {source}, Chunk {chunk_id}]\n{content}\n"
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    def _generate_grounded_response(self, question: str, 
                                   context: str,
                                   relevant_docs: List[Dict[str, Any]],
                                   use_history: bool) -> Dict[str, Any]:
        """
        Generate a grounded response with citations
        
        Args:
            question: User question
            context: Context from retrieved documents
            relevant_docs: List of relevant documents
            use_history: Whether to use conversation history
            
        Returns:
            Response dictionary with answer and citations
        """
        # Build system prompt
        system_prompt = self._get_system_prompt()
        
        # Load user prompt template from file
        prompt_path = os.path.join('prompts', 'user_prompt_template.txt')
        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                user_prompt_template = f.read()
            user_prompt = user_prompt_template.format(context=context, question=question)
        except FileNotFoundError:
            # Fallback to inline prompt if file not found
            user_prompt = f"""Context from documents:
{context}

Question: {question}

INSTRUCTIONS:
- Answer the question ONLY using information from the context above
- Write in a direct, professional tone as if presenting established facts
- DO NOT reference "Document 1", "Document 2", or any document numbers in your answer
- DO NOT use phrases like "According to the documents" or "The document states"
- DO NOT use any external knowledge or information not present in the context
- If the answer is not in the context, respond with: "This information is not available in the provided documents."
- If only partial information is available, acknowledge what you found and clearly state what information is missing
- Present information confidently and naturally - the citation system will show sources separately

Answer:"""
        
        # Get conversation history if requested
        history = None
        if use_history:
            history = self.conversation_memory.get_conversation_history()
        
        # Create messages
        messages = self.llm_client.create_chat_messages(
            system_prompt=system_prompt,
            user_query=user_prompt,
            conversation_history=history
        )
        
        # Generate response
        answer = self.llm_client.generate_response(messages, temperature=0.3)
        
        # Check if LLM stated information is not available
        if "This information is not available in the provided documents" in answer:
            return {
                "answer": answer,
                "citations": [],
                "sources": [],
                "confidence": "low",
                "retrieved_docs": 0
            }
        
        # Extract citations from relevant documents
        citations = self._extract_citations(answer, relevant_docs)
        
        # Determine confidence based on relevance scores
        avg_score = sum(doc['score'] for doc in relevant_docs) / len(relevant_docs)
        confidence = "high" if avg_score > 0.7 else "medium" if avg_score > 0.5 else "low"
        
        return {
            "answer": answer,
            "citations": citations,
            "sources": self._get_unique_sources(relevant_docs),
            "confidence": confidence,
            "retrieved_docs": len(relevant_docs)
        }
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the LLM from external file"""
        prompt_path = os.path.join('prompts', 'system_prompt.txt')
        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            # Fallback to inline prompt if file not found
            return """You are a professional AI medical assistant specialized in document analysis and providing clear, evidence-based answers.

CRITICAL RULES (MUST FOLLOW):
1. YOU MUST ONLY answer using information explicitly stated in the provided document context below
2. If the answer or ANY PART of the answer is not found in the documents, YOU MUST respond with EXACTLY this phrase:
   "This information is not available in the provided documents."
3. DO NOT use external knowledge, medical training, or general information - ONLY use what is in the documents
4. DO NOT make assumptions or inferences beyond what is explicitly stated in the documents

Response Guidelines:
- Write in a professional, confident tone presenting information as established facts
- DO NOT mention document numbers or use attribution phrases
- Present information directly and naturally

REMEMBER: Accuracy and honesty are MORE IMPORTANT than providing an answer."""
    
    def _extract_citations(self, answer: str, 
                          relevant_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract citations from relevant documents
        
        Args:
            answer: Generated answer
            relevant_docs: List of relevant documents
            
        Returns:
            List of citation dictionaries with all retrieved chunks
        """
        citations = []
        
        for idx, doc in enumerate(relevant_docs, 1):
            metadata = doc['metadata']
            source = metadata.get('source', 'Unknown')
            
            citation = {
                "source": source,
                "chunk_id": metadata.get('chunk_id', 0),
                "source_type": metadata.get('source_type', 'uploaded'),
                "relevance_score": round(doc['score'], 3),
                "file_path": metadata.get('file_path', ''),
                "chunk_content": doc['content'],
                "citation_number": idx,
                "metadata": {
                    "gdrive_link": metadata.get('gdrive_link', '')  # Include Google Drive link
                }
            }
            citations.append(citation)
        
        return citations
    
    def _get_unique_sources(self, documents: List[Dict[str, Any]]) -> List[str]:
        """Get list of unique source documents"""
        sources = set()
        for doc in documents:
            sources.add(doc['metadata'].get('source', 'Unknown'))
        return sorted(list(sources))
