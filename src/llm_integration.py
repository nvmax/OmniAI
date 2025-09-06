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

# Import the new Gemini client
try:
    from .new_gemini_client import NewGeminiClient as GeminiClient
except ImportError:
    from new_gemini_client import NewGeminiClient as GeminiClient

# Keep the old class as a fallback (but it won't be used)
class OldGeminiClient:
    """Old client for communicating with Google Gemini API (deprecated)."""

    def __init__(self):
        self.api_key = config.gemini_api_key
        self.model_name = config.gemini_model
        self.temperature = config.gemini_temperature
        self.max_tokens = config.gemini_max_tokens
        self.client = None
        self.is_connected = False

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()

    async def connect(self):
        """Initialize the Gemini client using the new google-genai package."""
        try:
            if not self.api_key:
                raise ValueError("Gemini API key not configured")

            from google import genai
            import os

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

        # Remove excessive whitespace
        response = re.sub(r'\n\s*\n\s*\n', '\n\n', response)
        response = response.strip()

        return response

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
        """Generate a response using Gemini."""
        try:
            if not self.is_connected or not self.model:
                await self.connect()

            # Handle both parameter names for compatibility
            message = user_message or prompt
            if not message:
                raise ValueError("Either user_message or prompt must be provided")

            # Prepare the full prompt
            if system_message:
                full_prompt = f"{system_message}\n\nUser: {message}\n\nAssistant:"
            else:
                full_prompt = message

            # Set generation parameters
            generation_config = {
                "temperature": temperature or self.temperature,
                "max_output_tokens": max_tokens or self.max_tokens,
            }

            # Configure tools if needed
            tools = None
            if use_search:
                # Use the simple dictionary format for Google Search tool
                tools = [{"google_search": {}}]

            # Configure for structured output if requested
            if structured_output:
                generation_config["response_mime_type"] = "application/json"

            # Generate response with tools if specified
            try:
                if tools:
                    # Try with tools first
                    response = await asyncio.to_thread(
                        self.model.generate_content,
                        full_prompt,
                        generation_config=generation_config,
                        tools=tools
                    )
                else:
                    response = await asyncio.to_thread(
                        self.model.generate_content,
                        full_prompt,
                        generation_config=generation_config
                    )
            except Exception as tool_error:
                # If tools fail, fall back to regular generation
                if tools:
                    logger.warning(f"Tool usage failed, falling back to regular generation: {tool_error}")
                    response = await asyncio.to_thread(
                        self.model.generate_content,
                        full_prompt,
                        generation_config=generation_config
                    )
                else:
                    raise tool_error

            if response:
                # Handle different response formats
                try:
                    if hasattr(response, 'text') and response.text:
                        cleaned_response = self._clean_response(response.text)
                        logger.debug(f"Gemini response length: {len(cleaned_response)} characters")
                        return cleaned_response
                    elif hasattr(response, 'candidates') and response.candidates:
                        # Try to extract text from candidates
                        for candidate in response.candidates:
                            if hasattr(candidate, 'content') and candidate.content:
                                if hasattr(candidate.content, 'parts') and candidate.content.parts:
                                    for part in candidate.content.parts:
                                        if hasattr(part, 'text') and part.text:
                                            cleaned_response = self._clean_response(part.text)
                                            logger.debug(f"Gemini response from parts: {len(cleaned_response)} characters")
                                            return cleaned_response

                    logger.warning(f"Gemini response format not recognized. Response: {response}")
                    return None

                except Exception as parse_error:
                    logger.error(f"Error parsing Gemini response: {parse_error}")
                    return None
            else:
                logger.warning("Gemini returned no response")
                return None

        except Exception as e:
            logger.error(f"Error generating Gemini response: {e}")
            return None

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> Optional[str]:
        """
        Generate a chat completion using Gemini.
        Converts OpenAI-style messages to Gemini format.
        """
        try:
            if not self.is_connected or not self.model:
                await self.connect()

            # Convert messages to a single prompt
            prompt_parts = []
            system_message = ""

            for message in messages:
                role = message.get("role", "")
                content = message.get("content", "")

                if role == "system":
                    system_message = content
                elif role == "user":
                    prompt_parts.append(f"User: {content}")
                elif role == "assistant":
                    prompt_parts.append(f"Assistant: {content}")

            # Combine into full prompt
            conversation = "\n\n".join(prompt_parts)
            if system_message:
                full_prompt = f"{system_message}\n\n{conversation}\n\nAssistant:"
            else:
                full_prompt = f"{conversation}\n\nAssistant:"

            # Generate response
            return await self.generate_response(
                full_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )

        except Exception as e:
            logger.error(f"Error in Gemini chat completion: {e}")
            return None

    async def search_and_respond(
        self,
        query: str,
        system_message: str = "",
        max_tokens: Optional[int] = None
    ) -> Optional[str]:
        """Use Gemini with Google Search to answer queries requiring current information."""
        try:
            if not self.is_connected or not self.model:
                await self.connect()

            # Create search-optimized prompt
            search_prompt = f"""Please search for current information about: {query}

Use the search results to provide an accurate, up-to-date response. If you find relevant information, cite it appropriately."""

            if system_message:
                search_prompt = f"{system_message}\n\n{search_prompt}"

            # Use Google Search tool with simple dictionary format
            tools = [{"google_search": {}}]

            generation_config = {
                "temperature": 0.3,  # Lower temperature for factual queries
                "max_output_tokens": max_tokens or self.max_tokens,
            }

            response = await asyncio.to_thread(
                self.model.generate_content,
                search_prompt,
                generation_config=generation_config,
                tools=tools
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

    async def generate_structured_response(
        self,
        prompt: str,
        schema: dict,
        system_message: str = "",
        max_tokens: Optional[int] = None
    ) -> Optional[dict]:
        """Generate a structured JSON response following a specific schema."""
        try:
            if not self.is_connected or not self.model:
                await self.connect()

            # Create structured prompt
            structured_prompt = f"""{system_message}

{prompt}

Please respond with valid JSON following this schema:
{schema}"""

            generation_config = {
                "temperature": 0.1,  # Very low temperature for structured output
                "max_output_tokens": max_tokens or self.max_tokens,
                "response_mime_type": "application/json"
            }

            response = await asyncio.to_thread(
                self.model.generate_content,
                structured_prompt,
                generation_config=generation_config
            )

            if response and response.text:
                try:
                    import json
                    return json.loads(response.text)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse structured response: {e}")
                    return None
            else:
                logger.warning("Gemini structured response returned empty")
                return None

        except Exception as e:
            logger.error(f"Error in Gemini structured response: {e}")
            return None

class LLMProviderFactory:
    """Factory for creating and managing LLM providers."""

    @staticmethod
    async def create_provider(provider_type: str = None):
        """Create an LLM provider based on configuration or type."""
        if provider_type is None:
            provider_type = config.llm_provider

        if provider_type == "gemini":
            logger.info("Creating Gemini provider (no fallback)")
            return GeminiClient()
        elif provider_type == "lm_studio":
            logger.info("Creating LM Studio provider (no fallback)")
            return LMStudioClient()
        elif provider_type == "auto":
            # For auto mode, try Gemini first, then LM Studio, but return the first working one
            logger.info("Auto-selecting provider...")

            if config.gemini_api_key:
                try:
                    gemini_client = GeminiClient()
                    await gemini_client.connect()
                    if await gemini_client.health_check():
                        logger.info("Auto-selected Gemini provider")
                        return gemini_client
                    else:
                        await gemini_client.disconnect()
                except Exception as e:
                    logger.debug(f"Gemini auto-selection failed: {e}")

            # Try LM Studio
            try:
                lm_client = LMStudioClient()
                await lm_client.connect()
                if await lm_client.health_check():
                    logger.info("Auto-selected LM Studio provider")
                    return lm_client
                else:
                    await lm_client.disconnect()
            except Exception as e:
                logger.debug(f"LM Studio auto-selection failed: {e}")

            # If both fail, raise an error instead of defaulting
            raise RuntimeError("Auto-selection failed: No working providers available")
        else:
            raise ValueError(f"Unknown LLM provider: {provider_type}")

    @staticmethod
    async def get_available_providers() -> List[str]:
        """Get list of available and working providers."""
        available = []

        # Test Gemini
        if config.gemini_api_key:
            try:
                gemini_client = GeminiClient()
                await gemini_client.connect()
                if await gemini_client.health_check():
                    available.append("gemini")
                await gemini_client.disconnect()
            except Exception:
                pass

        # Test LM Studio
        try:
            lm_client = LMStudioClient()
            await lm_client.connect()
            if await lm_client.health_check():
                available.append("lm_studio")
            await lm_client.disconnect()
        except Exception:
            pass

        return available

class LLMManager:
    """High-level manager for LLM operations with multi-provider support and automatic fallback."""

    def __init__(self):
        self.primary_client = None
        self.fallback_client = None
        self.current_provider = None
        self.max_retries = 3
        self.retry_delay = 1.0
        self._initialized = False

    async def _initialize_providers(self):
        """Initialize LLM providers based on configuration - no fallback."""
        if self._initialized:
            return

        try:
            # Get the specified provider only (no fallback setup)
            provider_type = config.llm_provider

            if provider_type == "auto":
                # For auto mode, still try to set up both but prefer one
                self.primary_client = await LLMProviderFactory.create_provider("auto")
                await self.primary_client.connect()

                if isinstance(self.primary_client, GeminiClient):
                    self.current_provider = "gemini"
                else:
                    self.current_provider = "lm_studio"

                # No fallback client for auto mode either
                self.fallback_client = None

            else:
                # For specific provider, use only that provider
                self.primary_client = await LLMProviderFactory.create_provider(provider_type)
                await self.primary_client.connect()
                self.current_provider = provider_type
                self.fallback_client = None  # No fallback

            self._initialized = True
            logger.info(f"LLM Manager initialized with provider: {self.current_provider} (no fallback)")

        except Exception as e:
            logger.error(f"Failed to initialize LLM provider '{config.llm_provider}': {e}")
            raise  # Don't create fallback, let it fail
    
    async def generate_response(
        self,
        user_message: str,
        system_message: str = "",
        conversation_history: List[Dict[str, str]] = None,
        use_search: bool = False,
        **kwargs
    ) -> Optional[str]:
        """Generate response using only the specified provider (no fallback)."""

        # Initialize providers if not done yet
        await self._initialize_providers()

        # Use only the primary provider - no fallback
        response = await self._try_provider(
            self.primary_client,
            self.current_provider,
            user_message,
            system_message,
            conversation_history,
            use_search=use_search,
            **kwargs
        )

        if response:
            return response

        # No fallback - return error message specific to the configured provider
        return f"I'm experiencing technical difficulties with {self.current_provider}. Please check the {self.current_provider} configuration and try again."

    async def _try_provider(
        self,
        client,
        provider_name: str,
        user_message: str,
        system_message: str = "",
        conversation_history: List[Dict[str, str]] = None,
        **kwargs
    ) -> Optional[str]:
        """Try to generate response with a specific provider."""

        for attempt in range(self.max_retries):
            try:
                # Check health before attempting
                if not await client.health_check():
                    logger.warning(f"{provider_name} provider health check failed on attempt {attempt + 1}")
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(self.retry_delay * (attempt + 1))
                        continue
                    else:
                        break

                response = await client.generate_response(
                    user_message=user_message,
                    system_message=system_message,
                    conversation_history=conversation_history,
                    **kwargs
                )

                if response:
                    logger.debug(f"Successful response from {provider_name} provider")
                    return response

                logger.warning(f"Empty response from {provider_name} provider on attempt {attempt + 1}")

            except Exception as e:
                logger.error(f"Error with {provider_name} provider on attempt {attempt + 1}: {e}")

            if attempt < self.max_retries - 1:
                await asyncio.sleep(self.retry_delay * (attempt + 1))

        logger.warning(f"{provider_name} provider failed after {self.max_retries} attempts")
        return None

    async def health_check(self) -> bool:
        """Check if the specified LLM provider is accessible."""
        await self._initialize_providers()

        try:
            # Check only the primary (specified) provider
            if self.primary_client:
                return await self.primary_client.health_check()

            return False

        except Exception as e:
            logger.error(f"Health check error for {self.current_provider}: {e}")
            return False

    async def close(self):
        """Close the LLM client connection."""
        try:
            if self.primary_client:
                await self.primary_client.disconnect()
        except Exception as e:
            logger.error(f"Error closing LLM connection: {e}")

    async def get_provider_status(self) -> Dict[str, Any]:
        """Get status of the specified provider only."""
        await self._initialize_providers()

        status = {
            "current_provider": self.current_provider,
            "configured_provider": config.llm_provider,
            "fallback_enabled": False,
            "providers": {}
        }

        # Check only the primary (specified) provider
        if self.primary_client:
            try:
                primary_healthy = await self.primary_client.health_check()
                status["providers"][self.current_provider] = {
                    "status": "healthy" if primary_healthy else "unhealthy",
                    "role": "primary",
                    "model": getattr(self.primary_client, 'model_name', 'unknown')
                }
            except Exception as e:
                status["providers"][self.current_provider] = {
                    "status": "error",
                    "role": "primary",
                    "error": str(e),
                    "model": getattr(self.primary_client, 'model_name', 'unknown')
                }

        return status

# Global LLM manager instance
llm_manager = LLMManager()
