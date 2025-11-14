"""
Conversation Memory Module
Manages conversation history and context
"""
from typing import List, Dict, Any
from datetime import datetime


class ConversationMemory:
    """Manages conversation history for multi-turn conversations"""
    
    def __init__(self, max_history: int = 10):
        """
        Initialize conversation memory
        
        Args:
            max_history: Maximum number of conversation turns to keep
        """
        self.max_history = max_history
        self.conversation_history = []
        self.metadata = {
            "session_start": datetime.now().isoformat(),
            "total_turns": 0
        }
    
    def add_user_message(self, message: str) -> None:
        """
        Add a user message to history
        
        Args:
            message: User's message
        """
        self.conversation_history.append({
            "role": "user",
            "content": message,
            "timestamp": datetime.now().isoformat()
        })
        self._trim_history()
    
    def add_assistant_message(self, message: str, 
                             citations: List[Dict[str, Any]] = None) -> None:
        """
        Add an assistant message to history
        
        Args:
            message: Assistant's response
            citations: List of citation information
        """
        entry = {
            "role": "assistant",
            "content": message,
            "timestamp": datetime.now().isoformat()
        }
        
        if citations:
            entry["citations"] = citations
        
        self.conversation_history.append(entry)
        self.metadata["total_turns"] += 1
        self._trim_history()
    
    def get_conversation_history(self, 
                                include_metadata: bool = False) -> List[Dict[str, str]]:
        """
        Get conversation history formatted for LLM
        
        Args:
            include_metadata: Whether to include timestamps and citations
            
        Returns:
            List of conversation messages
        """
        if include_metadata:
            return self.conversation_history
        
        # Return simplified format for LLM
        return [
            {"role": msg["role"], "content": msg["content"]}
            for msg in self.conversation_history
        ]
    
    def clear_history(self) -> None:
        """Clear all conversation history"""
        self.conversation_history = []
        self.metadata = {
            "session_start": datetime.now().isoformat(),
            "total_turns": 0
        }
    
    def _trim_history(self) -> None:
        """Trim history to maximum length"""
        if len(self.conversation_history) > self.max_history * 2:
            # Keep only recent messages (user + assistant pairs)
            self.conversation_history = self.conversation_history[-(self.max_history * 2):]
