"""
Image Analysis Agent - Analyzes uploaded images and provides intelligent commentary.
Uses vision transformer models to understand image content and LLM for detailed feedback.
"""

import logging
import asyncio
import aiohttp
import io
from typing import Dict, List, Optional, Any
from PIL import Image
import base64

logger = logging.getLogger(__name__)

class ImageAnalysisAgent:
    """Agent for analyzing images and providing intelligent commentary."""
    
    def __init__(self):
        self.vision_model = None
        self.processor = None
        self.model_loaded = False
        self._initialize_vision_model()
    
    def _initialize_vision_model(self):
        """Initialize the vision transformer model."""
        try:
            # Try to import transformers for vision models
            from transformers import BlipProcessor, BlipForConditionalGeneration
            import torch
            
            logger.info("Loading BLIP vision model for image analysis...")
            
            # Use BLIP model for image captioning and analysis
            model_name = "Salesforce/blip-image-captioning-base"
            
            self.processor = BlipProcessor.from_pretrained(model_name)
            self.vision_model = BlipForConditionalGeneration.from_pretrained(model_name)
            
            # Move to GPU if available
            if torch.cuda.is_available():
                self.vision_model = self.vision_model.cuda()
                logger.info("Vision model loaded on GPU")
            else:
                logger.info("Vision model loaded on CPU")
            
            self.model_loaded = True
            logger.info("Image analysis agent initialized successfully")
            
        except ImportError:
            logger.warning("Transformers library not available. Image analysis will use fallback method.")
            self.model_loaded = False
        except Exception as e:
            logger.error(f"Failed to load vision model: {e}")
            logger.info("Image analysis will use fallback method")
            self.model_loaded = False
    
    async def analyze_image(self, image_url: str, user_id: str, context: Dict = None) -> str:
        """Analyze an image and provide intelligent commentary."""
        try:
            logger.info(f"Analyzing image for user {user_id}: {image_url}")
            
            # Download the image
            image = await self._download_image(image_url)
            if not image:
                return "I couldn't download the image. Please make sure the image is accessible."
            
            # Analyze the image content
            if self.model_loaded:
                analysis = await self._analyze_with_vision_model(image)
            else:
                analysis = await self._analyze_with_fallback(image_url)
            
            # Generate intelligent commentary using LLM
            commentary = await self._generate_commentary(analysis, user_id, context)
            
            return commentary
            
        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            return f"I encountered an error while analyzing the image: {str(e)}"
    
    async def _download_image(self, image_url: str) -> Optional[Image.Image]:
        """Download image from URL."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(image_url) as response:
                    if response.status == 200:
                        image_data = await response.read()
                        image = Image.open(io.BytesIO(image_data))
                        
                        # Convert to RGB if necessary
                        if image.mode != 'RGB':
                            image = image.convert('RGB')
                        
                        logger.info(f"Downloaded image: {image.size} pixels, mode: {image.mode}")
                        return image
                    else:
                        logger.error(f"Failed to download image: HTTP {response.status}")
                        return None
                        
        except Exception as e:
            logger.error(f"Error downloading image: {e}")
            return None
    
    async def _analyze_with_vision_model(self, image: Image.Image) -> Dict[str, Any]:
        """Analyze image using vision transformer model."""
        try:
            import torch
            
            # Process the image
            inputs = self.processor(image, return_tensors="pt")
            
            # Move to GPU if available
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Generate caption
            with torch.no_grad():
                out = self.vision_model.generate(**inputs, max_length=50, num_beams=5)
            
            caption = self.processor.decode(out[0], skip_special_tokens=True)
            
            # Get image properties
            width, height = image.size
            aspect_ratio = width / height
            
            analysis = {
                "caption": caption,
                "dimensions": f"{width}x{height}",
                "aspect_ratio": round(aspect_ratio, 2),
                "orientation": "landscape" if aspect_ratio > 1.2 else "portrait" if aspect_ratio < 0.8 else "square",
                "analysis_method": "vision_transformer"
            }
            
            logger.info(f"Vision model analysis: {caption}")
            return analysis
            
        except Exception as e:
            logger.error(f"Error in vision model analysis: {e}")
            return await self._analyze_with_fallback_basic(image)
    
    async def _analyze_with_fallback(self, image_url: str) -> Dict[str, Any]:
        """Fallback analysis when vision model is not available."""
        return {
            "caption": "Image uploaded by user",
            "dimensions": "Unknown",
            "aspect_ratio": "Unknown",
            "orientation": "Unknown",
            "analysis_method": "fallback",
            "note": "Vision model not available - using basic analysis"
        }
    
    async def _analyze_with_fallback_basic(self, image: Image.Image) -> Dict[str, Any]:
        """Basic fallback analysis with image properties."""
        width, height = image.size
        aspect_ratio = width / height
        
        return {
            "caption": "Image content analysis not available",
            "dimensions": f"{width}x{height}",
            "aspect_ratio": round(aspect_ratio, 2),
            "orientation": "landscape" if aspect_ratio > 1.2 else "portrait" if aspect_ratio < 0.8 else "square",
            "analysis_method": "basic_fallback"
        }
    
    async def _generate_commentary(self, analysis: Dict[str, Any], user_id: str, context: Dict = None) -> str:
        """Generate intelligent commentary using LLM."""
        try:
            # Import LLM manager
            try:
                from ..llm_integration import llm_manager
                from ..personalities import personality_manager
            except ImportError:
                from llm_integration import llm_manager
                from personalities import personality_manager
            
            # Get user's personality
            personality = personality_manager.get_user_personality(user_id)
            
            # Create prompt for LLM
            prompt = self._create_analysis_prompt(analysis)
            
            # Generate commentary
            enhanced_system_message = f"""{personality.system_message}

You are analyzing an image. Provide insightful, engaging commentary about what you see.

IMPORTANT RULES:
- Do NOT use any thinking tags like <think>, <thinking>, <thought>, etc.
- Do NOT show your internal reasoning process
- Provide only the final commentary
- Be direct and engaging
- Keep response under 300 words"""

            commentary = await llm_manager.generate_response(
                user_message=prompt,
                system_message=enhanced_system_message,
                max_tokens=300
            )

            if commentary:
                # Clean the commentary to remove thinking tags
                cleaned_commentary = self._clean_commentary(commentary)
                return cleaned_commentary
            else:
                return self._create_fallback_commentary(analysis)
                
        except Exception as e:
            logger.error(f"Error generating commentary: {e}")
            return self._create_fallback_commentary(analysis)
    
    def _create_analysis_prompt(self, analysis: Dict[str, Any]) -> str:
        """Create a prompt for LLM analysis."""
        caption = analysis.get("caption", "Unknown content")
        dimensions = analysis.get("dimensions", "Unknown")
        orientation = analysis.get("orientation", "Unknown")
        method = analysis.get("analysis_method", "Unknown")
        
        prompt = f"""I've analyzed an image and here's what I found:

**Image Content:** {caption}
**Dimensions:** {dimensions}
**Orientation:** {orientation}
**Analysis Method:** {method}

Please provide an engaging commentary about this image. Consider:
- What's interesting or notable about the content
- The composition and visual elements
- Any artistic or technical aspects worth mentioning
- Keep it conversational and insightful

Provide your commentary:"""
        
        return prompt

    def _clean_commentary(self, commentary: str) -> str:
        """Clean commentary to remove thinking tags and unwanted elements."""
        import re

        if not commentary:
            return commentary

        # Remove thinking tags and their content (case insensitive, multiline)
        commentary = re.sub(r'<think>.*?</think>', '', commentary, flags=re.DOTALL | re.IGNORECASE)
        commentary = re.sub(r'<thinking>.*?</thinking>', '', commentary, flags=re.DOTALL | re.IGNORECASE)
        commentary = re.sub(r'<thought>.*?</thought>', '', commentary, flags=re.DOTALL | re.IGNORECASE)
        commentary = re.sub(r'<internal>.*?</internal>', '', commentary, flags=re.DOTALL | re.IGNORECASE)
        commentary = re.sub(r'<reasoning>.*?</reasoning>', '', commentary, flags=re.DOTALL | re.IGNORECASE)

        # Remove any remaining XML-like tags that might contain thinking
        commentary = re.sub(r'<[^>]*thinking[^>]*>.*?</[^>]*>', '', commentary, flags=re.DOTALL | re.IGNORECASE)

        # More aggressive cleaning - remove anything that looks like thinking
        # Remove lines that start with thinking patterns
        lines = commentary.split('\n')
        cleaned_lines = []

        for line in lines:
            line = line.strip()
            # Skip lines that look like thinking patterns
            if any(pattern in line.lower() for pattern in ['<think', 'okay, let me', 'first, i need', 'now, considering', 'the user wants']):
                continue
            # Skip empty lines
            if not line:
                continue
            cleaned_lines.append(line)

        commentary = '\n'.join(cleaned_lines)

        # Remove duplicate consecutive lines
        lines = commentary.split('\n')
        final_lines = []
        prev_line = None

        for line in lines:
            line = line.strip()
            if line != prev_line and line:  # Only add if different from previous and not empty
                final_lines.append(line)
                prev_line = line

        commentary = '\n'.join(final_lines)

        # Clean up excessive whitespace
        commentary = re.sub(r'\n\s*\n\s*\n+', '\n\n', commentary)  # Multiple newlines to double
        commentary = commentary.strip()

        # If commentary is empty or too short after cleaning, provide fallback
        if len(commentary) < 20:
            return "I can see this image, but I'm having trouble generating proper commentary right now."

        return commentary

    def _create_fallback_commentary(self, analysis: Dict[str, Any]) -> str:
        """Create basic commentary when LLM is not available."""
        caption = analysis.get("caption", "Image content")
        dimensions = analysis.get("dimensions", "Unknown size")
        orientation = analysis.get("orientation", "Unknown orientation")
        
        return f"""**Image Analysis:**

📸 **Content:** {caption}
📏 **Size:** {dimensions}
🖼️ **Format:** {orientation}

I can see this is an interesting image! The {orientation} orientation works well for the composition."""

class ImageAnalysisExecutor:
    """Executor for image analysis tasks."""
    
    def __init__(self):
        self.agent = ImageAnalysisAgent()
    
    async def analyze_image(self, image_url: str, user_id: str, context: Dict = None) -> str:
        """Execute image analysis task."""
        try:
            return await self.agent.analyze_image(image_url, user_id, context)
        except Exception as e:
            logger.error(f"Image analysis execution failed: {e}")
            return f"Image analysis failed: {str(e)}"

# Global image analysis executor instance
image_analysis_executor = ImageAnalysisExecutor()
