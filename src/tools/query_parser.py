"""
Query Parser for extracting search subjects from natural language queries.
Uses HuggingFace models for intelligent query parsing with regex fallback.
"""

import logging
import re
from typing import Optional, Dict, Any
import asyncio

logger = logging.getLogger(__name__)

class QueryParser:
    """Intelligent query parser using HuggingFace models with regex fallback."""
    
    def __init__(self):
        self.hf_model = None
        self.hf_tokenizer = None
        self.model_loaded = False
        # Disable model loading for now - regex works perfectly
        # self._load_model()
    
    def _load_model(self):
        """Load the HuggingFace query parser model."""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch

            # Try different models in order of preference
            model_options = [
                "microsoft/DialoGPT-small",  # Smaller, more reliable model
                "facebook/blenderbot-400M-distill",  # Alternative option
                "EmbeddingStudio/query-parser-falcon-7b-instruct"  # Original choice
            ]

            for model_name in model_options:
                try:
                    logger.info(f"Attempting to load query parser model: {model_name}")

                    # Load tokenizer and model
                    self.hf_tokenizer = AutoTokenizer.from_pretrained(
                        model_name,
                        trust_remote_code=True
                    )
                    self.hf_model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                        device_map="auto" if torch.cuda.is_available() else None,
                        trust_remote_code=True
                    )

                    # Set pad token if not set
                    if self.hf_tokenizer.pad_token is None:
                        self.hf_tokenizer.pad_token = self.hf_tokenizer.eos_token

                    self.model_loaded = True
                    logger.info(f"Query parser model loaded successfully: {model_name}")
                    return

                except Exception as model_error:
                    logger.warning(f"Failed to load {model_name}: {model_error}")
                    continue

            # If all models failed
            raise Exception("All model options failed to load")

        except ImportError:
            logger.warning("Transformers library not available. Using regex fallback for query parsing.")
            self.model_loaded = False
        except Exception as e:
            logger.error(f"Failed to load any query parser model: {e}")
            logger.info("Falling back to regex-based query parsing")
            self.model_loaded = False
    
    def parse_image_query(self, query: str) -> Dict[str, Any]:
        """Parse an image search query to extract the subject and intent."""
        if self.model_loaded:
            return self._parse_with_model(query)
        else:
            return self._parse_with_regex(query)
    
    def _parse_with_model(self, query: str) -> Dict[str, Any]:
        """Parse query using the HuggingFace model."""
        try:
            # Create a prompt for the query parser
            prompt = f"""Parse this image search query and extract the main subject:

Query: "{query}"

Extract:
1. Main subject (what the user wants to see)
2. Intent (image search)
3. Modifiers (any specific requirements)

Subject:"""

            # Tokenize and generate
            inputs = self.hf_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

            import torch
            with torch.no_grad():
                outputs = self.hf_model.generate(
                    **inputs,
                    max_new_tokens=50,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=self.hf_tokenizer.eos_token_id
                )
            
            # Decode the response
            response = self.hf_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract the subject from the response
            subject = self._extract_subject_from_response(response, query)
            
            return {
                "subject": subject,
                "intent": "image_search",
                "confidence": 0.9,
                "method": "model",
                "original_query": query
            }
            
        except Exception as e:
            logger.error(f"Model parsing failed: {e}")
            return self._parse_with_regex(query)
    
    def _extract_subject_from_response(self, response: str, original_query: str) -> str:
        """Extract the subject from the model's response."""
        try:
            # Look for the subject after "Subject:" in the response
            lines = response.split('\n')
            for line in lines:
                if 'subject:' in line.lower():
                    subject = line.split(':', 1)[1].strip()
                    if subject and len(subject) > 1:
                        return subject
            
            # If no clear subject found, try to extract from the end of response
            response_parts = response.split("Subject:")
            if len(response_parts) > 1:
                subject = response_parts[-1].strip().split('\n')[0].strip()
                if subject and len(subject) > 1:
                    return subject
            
            # Fallback to regex if model response is unclear
            return self._extract_subject_regex(original_query)
            
        except Exception as e:
            logger.error(f"Error extracting subject from model response: {e}")
            return self._extract_subject_regex(original_query)
    
    def _parse_with_regex(self, query: str) -> Dict[str, Any]:
        """Parse query using regex patterns as fallback."""
        subject = self._extract_subject_regex(query)
        
        return {
            "subject": subject,
            "intent": "image_search",
            "confidence": 0.7,
            "method": "regex",
            "original_query": query
        }
    
    def _extract_subject_regex(self, query: str) -> str:
        """Extract the actual subject from image search queries using regex."""
        query_lower = query.lower().strip()
        
        # Common image request patterns to remove
        patterns_to_remove = [
            r'^find\s+(?:me\s+)?(?:a\s+|an\s+)?(?:picture|image|photo|pic)\s+(?:of\s+)?',
            r'^show\s+(?:me\s+)?(?:a\s+|an\s+)?(?:picture|image|photo|pic)\s+(?:of\s+)?',
            r'^search\s+(?:for\s+)?(?:a\s+|an\s+)?(?:picture|image|photo|pic)\s+(?:of\s+)?',
            r'^get\s+(?:me\s+)?(?:a\s+|an\s+)?(?:picture|image|photo|pic)\s+(?:of\s+)?',
            r'^look\s+(?:for\s+)?(?:a\s+|an\s+)?(?:picture|image|photo|pic)\s+(?:of\s+)?',
            r'^(?:can\s+you\s+)?(?:find|search|get|show)\s+(?:me\s+)?(?:a\s+|an\s+)?(?:picture|image|photo|pic)\s+(?:of\s+)?',
            r'(?:picture|image|photo|pic)\s+(?:of\s+)?',
            r'(?:and\s+)?(?:give\s+it\s+to\s+me|show\s+it\s+to\s+me).*$'
        ]
        
        # Apply patterns to extract the subject
        cleaned_query = query_lower
        for pattern in patterns_to_remove:
            cleaned_query = re.sub(pattern, '', cleaned_query).strip()
        
        # Remove common filler words at the beginning
        filler_words = ['the', 'a', 'an', 'some', 'any']
        words = cleaned_query.split()
        while words and words[0] in filler_words:
            words.pop(0)
        
        cleaned_query = ' '.join(words).strip()
        
        # If we removed too much, fall back to original query
        if len(cleaned_query) < 2:
            # Try a simpler extraction - look for "of X" pattern
            of_match = re.search(r'\bof\s+(.+?)(?:\s+and\s+give|$)', query_lower)
            if of_match:
                cleaned_query = of_match.group(1).strip()
            else:
                cleaned_query = query.strip()
        
        return cleaned_query

# Global query parser instance
query_parser = QueryParser()
