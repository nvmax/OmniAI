"""
LM Studio integration for Omni-Assistant Discord Bot.
Handles communication with local LM Studio server for LLM inference.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any
import aiohttp
from aiohttp import ClientTimeout, ClientError

try:
    from .config import config
except ImportError:
    from config import config

logger = logging.getLogger(__name__)

class LMStudioClient:
    """Client for communicating with LM Studio local server."""
    
    def __init__(self):
        self.base_url = config.lm_studio_url
        self.chat_url = config.lm_studio_chat_url
        self.model = config.lm_studio_model
        self.session: Optional[aiohttp.ClientSession] = None
        self.timeout = ClientTimeout(total=120)  # 2 minute timeout
        
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()
    
    async def connect(self):
        """Initialize the HTTP session."""
        if not self.session or self.session.closed:
            self.session = aiohttp.ClientSession(timeout=self.timeout)
    
    async def disconnect(self):
        """Close the HTTP session."""
        if self.session and not self.session.closed:
            await self.session.close()

    def _clean_response(self, response: str) -> str:
        """Clean up the response by removing unwanted elements."""
        import re

        if not response:
            return response

        # AGGRESSIVE THINKING TAG REMOVAL
        # Remove thinking tags and their content (case insensitive, multiline)
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL | re.IGNORECASE)
        response = re.sub(r'<thinking>.*?</thinking>', '', response, flags=re.DOTALL | re.IGNORECASE)
        response = re.sub(r'<thought>.*?</thought>', '', response, flags=re.DOTALL | re.IGNORECASE)
        response = re.sub(r'<internal>.*?</internal>', '', response, flags=re.DOTALL | re.IGNORECASE)
        response = re.sub(r'<reasoning>.*?</reasoning>', '', response, flags=re.DOTALL | re.IGNORECASE)

        # Remove incomplete thinking tags (in case they're cut off)
        response = re.sub(r'<think[^>]*$', '', response, flags=re.MULTILINE | re.IGNORECASE)
        response = re.sub(r'^[^<]*</think>', '', response, flags=re.MULTILINE | re.IGNORECASE)

        # Split into lines and aggressively filter
        lines = response.split('\n')
        cleaned_lines = []

        skip_until_end = False

        for line in lines:
            line = line.strip()

            # Skip lines that contain thinking patterns
            if any(pattern in line.lower() for pattern in [
                '<think', 'okay, let me', 'first, i need', 'now, considering',
                'the user wants', 'let me start by', 'i need to make sure',
                'maybe point out', 'composition-wise', 'artistic aspects',
                'need to keep it', 'technical aspects'
            ]):
                skip_until_end = True
                continue

            # If we hit a proper response start, stop skipping
            if line and not skip_until_end:
                cleaned_lines.append(line)
            elif line and skip_until_end:
                # Check if this looks like actual content (not thinking)
                if not any(thinking_word in line.lower() for thinking_word in [
                    'maybe', 'i can', 'i should', 'perhaps', 'let me', 'first', 'next'
                ]):
                    skip_until_end = False
                    cleaned_lines.append(line)

        # If we have cleaned lines, use them; otherwise try to salvage something
        if cleaned_lines:
            response = '\n'.join(cleaned_lines)
        else:
            # Last resort: try to find content after the last thinking tag
            last_think_end = response.rfind('</think>')
            if last_think_end != -1:
                response = response[last_think_end + 8:].strip()

        # Remove repeated identical lines
        lines = response.split('\n')
        final_lines = []
        prev_line = None

        for line in lines:
            line = line.strip()
            if line and line != prev_line:
                final_lines.append(line)
                prev_line = line

        response = '\n'.join(final_lines)

        # Remove excessive whitespace and clean up
        response = re.sub(r'\n\s*\n\s*\n+', '\n\n', response)  # Multiple newlines to double
        response = response.strip()

        # NUCLEAR OPTION: If response still contains thinking patterns, extract only the good part
        if '<think' in response.lower() or any(pattern in response.lower() for pattern in [
            'okay, so i need', 'let me start by', 'first, the color', 'composition-wise'
        ]):
            # Find the last occurrence of the filename (this usually marks the start of clean content)
            lines = response.split('\n')
            clean_start_idx = -1

            for i, line in enumerate(lines):
                # Look for lines that start the actual commentary (not thinking)
                if line.strip() and not any(bad_pattern in line.lower() for bad_pattern in [
                    '<think', 'okay,', 'let me', 'first,', 'composition-wise', 'artistic aspects',
                    'technical aspects', 'i need to', 'maybe', 'wait,', 'since i don'
                ]):
                    # This looks like actual content
                    if len(line.strip()) > 20:  # Make sure it's substantial
                        clean_start_idx = i
                        break

            if clean_start_idx != -1:
                response = '\n'.join(lines[clean_start_idx:])
            else:
                # Last resort: look for common angry response starters
                for i, line in enumerate(lines):
                    if any(starter in line.lower() for starter in [
                        'look at', 'what', 'this is', 'oh', 'ugh', 'seriously'
                    ]) and len(line.strip()) > 10:
                        response = '\n'.join(lines[i:])
                        break

        # Final cleanup
        response = response.strip()

        # If response is too short or empty after cleaning, provide fallback
        if len(response) < 10:
            return "I can see the content, but I'm having trouble generating a proper response right now."

        return response
    
    async def health_check(self) -> bool:
        """Check if LM Studio server is accessible."""
        try:
            await self.connect()
            async with self.session.get(f"{self.base_url}/v1/models") as response:
                return response.status == 200
        except Exception as e:
            logger.error(f"LM Studio health check failed: {e}")
            return False
    
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = None,
        stream: bool = False,
        **kwargs
    ) -> Optional[str]:
        """
        Send a chat completion request to LM Studio.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            **kwargs: Additional parameters for the API
        
        Returns:
            Generated response text or None if failed
        """
        # Use default max_tokens from config if not specified
        if max_tokens is None:
            max_tokens = config.default_max_tokens

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
            **kwargs
        }
        
        try:
            await self.connect()
            
            async with self.session.post(
                self.chat_url,
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"LM Studio API error {response.status}: {error_text}")
                    return None
                
                result = await response.json()
                
                if "choices" in result and len(result["choices"]) > 0:
                    response = result["choices"][0]["message"]["content"].strip()

                    # Log response length for debugging
                    logger.debug(f"Generated response length: {len(response)} characters")

                    # Check if response was truncated
                    choice = result["choices"][0]
                    if "finish_reason" in choice and choice["finish_reason"] == "length":
                        logger.warning("Response was truncated due to max_tokens limit")

                    # Clean up the response by removing thinking tags and other unwanted elements
                    response = self._clean_response(response)
                    return response
                else:
                    logger.error(f"Unexpected response format: {result}")
                    return None
                    
        except ClientError as e:
            logger.error(f"HTTP client error: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in chat completion: {e}")
            return None
    
    async def generate_response(
        self,
        user_message: str,
        system_message: str = "",
        conversation_history: List[Dict[str, str]] = None,
        **kwargs
    ) -> Optional[str]:
        """
        Generate a response using the LLM with conversation context.
        
        Args:
            user_message: The user's input message
            system_message: System prompt to guide the AI's behavior
            conversation_history: Previous messages in the conversation
            **kwargs: Additional parameters for chat completion
        
        Returns:
            Generated response or None if failed
        """
        messages = []
        
        # Add system message if provided
        if system_message:
            messages.append({"role": "system", "content": system_message})
        
        # Add conversation history if provided
        if conversation_history:
            messages.extend(conversation_history)
        
        # Add current user message
        messages.append({"role": "user", "content": user_message})
        
        return await self.chat_completion(messages, **kwargs)

class LLMManager:
    """High-level manager for LLM operations with retry logic and error handling."""
    
    def __init__(self):
        self.client = LMStudioClient()
        self.max_retries = 3
        self.retry_delay = 1.0
    
    async def generate_response(
        self,
        user_message: str,
        system_message: str = "",
        conversation_history: List[Dict[str, str]] = None,
        **kwargs
    ) -> Optional[str]:
        """Generate response with retry logic."""
        
        for attempt in range(self.max_retries):
            try:
                # Check health before attempting
                if not await self.client.health_check():
                    logger.warning(f"LM Studio health check failed on attempt {attempt + 1}")
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(self.retry_delay * (attempt + 1))
                        continue
                    else:
                        return "I'm having trouble connecting to my language model. Please check if LM Studio is running."
                
                response = await self.client.generate_response(
                    user_message=user_message,
                    system_message=system_message,
                    conversation_history=conversation_history,
                    **kwargs
                )
                
                if response:
                    return response
                
                logger.warning(f"Empty response on attempt {attempt + 1}")
                
            except Exception as e:
                logger.error(f"Error on attempt {attempt + 1}: {e}")
            
            if attempt < self.max_retries - 1:
                await asyncio.sleep(self.retry_delay * (attempt + 1))
        
        return "I'm experiencing technical difficulties. Please try again later."

    async def health_check(self) -> bool:
        """Check if LM Studio server is accessible."""
        try:
            return await self.client.health_check()
        except Exception as e:
            logger.error(f"Health check error: {e}")
            return False

    async def close(self):
        """Close the LLM client connection."""
        await self.client.disconnect()

# Global LLM manager instance
llm_manager = LLMManager()
