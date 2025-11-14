"""
LLM Client Module
Handles interaction with Euriai LLM API
"""
from typing import List, Dict, Any
from euriai import EuriaiClient
from config.config import Config


class LLMClient:
    """Client for interacting with Euriai LLM"""
    
    def __init__(self, api_key: str = None, model: str = None):
        """
        Initialize the LLM client
        
        Args:
            api_key: Euriai API key
            model: Model name to use
        """
        self.api_key = api_key or Config.EURIAI_API_KEY
        self.model = model or Config.EURIAI_MODEL
        
        if not self.api_key:
            raise ValueError("Euriai API key is required")
        
        self.client = EuriaiClient(
            api_key=self.api_key,
            model=self.model
        )
        print(f"LLM Client initialized with model: {self.model}")
    
    def generate_response(self, messages: List[Dict[str, str]], 
                         temperature: float = 0.3,
                         max_tokens: int = 2048) -> str:
        """
        Generate a response from the LLM
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens in response
            
        Returns:
            Generated text response
        """
        try:
            # Convert messages to Euriai format (single prompt string)
            prompt = self._messages_to_prompt(messages)
            
            response = self.client.generate_completion(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )

            return response["choices"][0]["message"]["content"]

        except Exception as e:
            err_text = str(e)
            print(f"Error generating response: {err_text}")
            return f"Error: Unable to generate response. {err_text}"
    
    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        Convert chat messages to a single prompt string
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Formatted prompt string
        """
        prompt_parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        return "\n\n".join(prompt_parts)
    
    def generate_streaming_response(self, messages: List[Dict[str, str]], 
                                   temperature: float = 0.3,
                                   max_tokens: int = 2048):
        """
        Generate a streaming response from the LLM
        Note: Euriai may not support streaming, so we'll return the full response
        
        Args:
            messages: List of message dictionaries
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            
        Yields:
            Chunks of generated text
        """
        try:
            # Since Euriai might not support streaming, we'll return the full response
            response = self.generate_response(messages, temperature, max_tokens)
            yield response
        
        except Exception as e:
            print(f"Error in streaming response: {str(e)}")
            yield f"Error: {str(e)}"
    
    def create_chat_messages(self, system_prompt: str, 
                           user_query: str,
                           conversation_history: List[Dict[str, str]] = None) -> List[Dict[str, str]]:
        """
        Create a properly formatted message list for chat
        
        Args:
            system_prompt: System instruction
            user_query: Current user query
            conversation_history: Previous conversation messages
            
        Returns:
            List of formatted messages
        """
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history if provided
        if conversation_history:
            messages.extend(conversation_history)
        
        # Add current user query
        messages.append({"role": "user", "content": user_query})
        
        return messages
