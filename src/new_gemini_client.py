"""
New Gemini client using the google-genai package with Google Search support.
"""

import asyncio
import logging
import os
from typing import Dict, List, Optional

try:
    from .config import config
except ImportError:
    from config import config

logger = logging.getLogger(__name__)

class NewGeminiClient:
    """Client for Google Gemini API using the new google-genai package with Google Search support."""
    
    def __init__(self):
        self.api_key = config.gemini_api_key
        self.model_name = config.gemini_model
        self.temperature = config.gemini_temperature
        self.max_tokens = config.gemini_max_tokens
        self.client = None
        self.is_connected = False

    async def connect(self):
        """Initialize the Gemini client using the new google-genai package."""
        try:
            if not self.api_key:
                raise ValueError("Gemini API key not configured")

            from google import genai
            
            # Set the API key as environment variable (required by new client)
            os.environ['GEMINI_API_KEY'] = self.api_key

            # Initialize the new client
            self.client = genai.Client()
            self.is_connected = True
            logger.info(f"Connected to Gemini model: {self.model_name}")

        except Exception as e:
            logger.error(f"Failed to connect to Gemini: {e}")
            self.is_connected = False
            raise

    async def disconnect(self):
        """Close the Gemini client connection."""
        self.client = None
        self.is_connected = False
        logger.info("Disconnected from Gemini")

    async def health_check(self) -> bool:
        """Check if Gemini API is accessible."""
        try:
            if not self.is_connected:
                await self.connect()

            if not self.client:
                return False

            # Simple test generation with a very basic prompt
            response = await self.generate_response("Hi", max_tokens=5)
            return bool(response and len(response.strip()) > 0)

        except Exception as e:
            logger.debug(f"Gemini health check failed: {e}")
            return False

    async def generate_response(
        self,
        user_message: str = None,
        prompt: str = None,
        system_message: str = "",
        conversation_history: List[Dict[str, str]] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        use_search: bool = False,
        structured_output: bool = False,
        **kwargs
    ) -> Optional[str]:
        """Generate a response using Gemini with the new google-genai package."""
        try:
            if not self.is_connected or not self.client:
                await self.connect()

            # Handle both parameter names for compatibility
            message = user_message or prompt
            if not message:
                raise ValueError("Either user_message or prompt must be provided")

            from google.genai import types

            # Prepare the full prompt
            if system_message:
                full_prompt = f"{system_message}\n\nUser: {message}\n\nAssistant:"
            else:
                full_prompt = message

            # Configure tools if needed
            tools = []
            if use_search:
                # Use the correct format for Google Search tool
                grounding_tool = types.Tool(
                    google_search=types.GoogleSearch()
                )
                tools.append(grounding_tool)
            
            # Set generation parameters
            config = types.GenerateContentConfig(
                temperature=temperature or self.temperature,
                max_output_tokens=max_tokens or self.max_tokens,
                tools=tools if tools else None
            )
            
            # Configure for structured output if requested
            if structured_output:
                config.response_mime_type = "application/json"

            # Generate response
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.model_name,
                contents=full_prompt,
                config=config
            )

            if response and response.text:
                cleaned_response = self._clean_response(response.text)
                logger.info(f"Gemini response length: {len(cleaned_response)} characters")
                return cleaned_response
            else:
                logger.warning("Gemini returned empty response")
                return None

        except Exception as e:
            logger.error(f"Error generating Gemini response: {e}")
            return None

    async def search_and_respond(
        self,
        query: str,
        system_message: str = "",
        max_tokens: Optional[int] = None
    ) -> Optional[str]:
        """Use Gemini with Google Search to answer queries requiring current information."""
        try:
            if not self.is_connected or not self.client:
                await self.connect()
            
            # Create search-optimized prompt
            search_prompt = f"""Please search for current information about: {query}

Use the search results to provide an accurate, up-to-date response. If you find relevant information, cite it appropriately."""
            
            if system_message:
                search_prompt = f"{system_message}\n\n{search_prompt}"
            
            from google.genai import types
            
            # Use Google Search tool
            grounding_tool = types.Tool(
                google_search=types.GoogleSearch()
            )
            
            config = types.GenerateContentConfig(
                temperature=0.3,  # Lower temperature for factual queries
                max_output_tokens=max_tokens or self.max_tokens,
                tools=[grounding_tool]
            )
            
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.model_name,
                contents=search_prompt,
                config=config
            )
            
            if response and response.text:
                cleaned_response = self._clean_response(response.text)
                logger.info(f"Gemini search response length: {len(cleaned_response)} characters")
                return cleaned_response
            else:
                logger.warning("Gemini search returned empty response")
                return None
                
        except Exception as e:
            logger.error(f"Error in Gemini search: {e}")
            return None

    def _clean_response(self, response: str) -> str:
        """Clean up the response by removing unwanted elements."""
        if not response:
            return response

        # Remove common unwanted patterns
        import re

        # Remove thinking tags and their content
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL | re.IGNORECASE)
        response = re.sub(r'<thinking>.*?</thinking>', '', response, flags=re.DOTALL | re.IGNORECASE)
        response = re.sub(r'<thought>.*?</thought>', '', response, flags=re.DOTALL | re.IGNORECASE)

        # Remove tool code blocks and attempts
        response = re.sub(r'```tool_code.*?```', '', response, flags=re.DOTALL | re.IGNORECASE)
        response = re.sub(r'```python.*?print\(goog.*?```', '', response, flags=re.DOTALL | re.IGNORECASE)
        response = re.sub(r'print\(goog[^)]*\)', '', response, flags=re.IGNORECASE)

        # Remove excessive whitespace
        response = re.sub(r'\n\s*\n\s*\n', '\n\n', response)
        response = response.strip()

        return response

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> Optional[str]:
        """Chat completion interface for compatibility."""
        try:
            # Convert messages to a single prompt
            prompt_parts = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "system":
                    prompt_parts.append(f"System: {content}")
                elif role == "user":
                    prompt_parts.append(f"User: {content}")
                elif role == "assistant":
                    prompt_parts.append(f"Assistant: {content}")
            
            full_prompt = "\n\n".join(prompt_parts) + "\n\nAssistant:"
            
            return await self.generate_response(
                prompt=full_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            
        except Exception as e:
            logger.error(f"Error in Gemini chat completion: {e}")
            return None
