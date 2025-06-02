"""
CrewAI orchestrator for managing multi-agent tasks and workflows.
Coordinates between different agents based on user requests and context.
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any
from crewai import Crew, Task

try:
    from .agents.base_agent import get_agent_for_task
    from .agents.research_agent import research_executor
    from .agents.coding_agent import coding_executor
    from .agents.web_search_agent import web_search_executor
    from .memory_manager import memory_manager
    from .personalities import personality_manager
    from .llm_integration import llm_manager
    from .config import config
except ImportError:
    from agents.base_agent import get_agent_for_task
    from agents.research_agent import research_executor
    from agents.coding_agent import coding_executor
    from agents.web_search_agent import web_search_executor
    from memory_manager import memory_manager
    from personalities import personality_manager
    from llm_integration import llm_manager
    from config import config

logger = logging.getLogger(__name__)

class TaskClassifier:
    """Classifies user requests to determine appropriate agent and workflow."""
    
    def __init__(self):
        self.task_patterns = {
            "image": [
                "picture", "image", "photo", "pic", "show me", "find me a picture",
                "find me an image", "search for a picture", "search for an image",
                "get me a pic", "get me a photo"
            ],
            "web_search": [
                "search for", "find", "lookup", "current", "latest", "recent",
                "stock price", "price of", "news about", "what's happening",
                "search the web", "google", "web search"
            ],
            "research": [
                "research", "information about", "detailed information", "explain in detail",
                "tell me everything about", "comprehensive", "in-depth", "analysis",
                "what is", "who is", "how does", "why does"
            ],
            "coding": [
                "code", "programming", "script", "function", "debug", "fix",
                "write code", "create function", "python", "javascript", "html"
            ],
            "general": [
                "help", "assist", "question", "chat", "talk", "discuss"
            ],
            "memory": [
                "remember", "recall", "what did we", "previous", "before",
                "conversation", "history"
            ],
            "personality": [
                "personality", "change mode", "be more", "act like", "switch to"
            ]
        }
    
    def classify_request(self, user_message: str) -> Dict[str, Any]:
        """Classify a user request and determine the appropriate response strategy."""
        message_lower = user_message.lower()

        classification = {
            "primary_task": "general",
            "secondary_tasks": [],
            "confidence": 0.0,
            "requires_memory": True,
            "requires_context": True,
            "complexity": "medium"
        }

        # Priority-based classification - check high-priority patterns first

        # 1. Check for image requests first (highest priority for simple requests)
        image_indicators = ["picture", "image", "photo", "pic"]
        if any(indicator in message_lower for indicator in image_indicators):
            classification["primary_task"] = "image"
            classification["confidence"] = 0.9
            classification["complexity"] = "low"
            return classification

        # 2. Check for simple web searches
        web_search_indicators = ["search for", "find", "lookup", "current", "latest", "recent", "stock price", "price of"]
        if any(indicator in message_lower for indicator in web_search_indicators):
            # But exclude if it's asking for detailed explanation
            if not any(research_word in message_lower for research_word in ["explain", "research", "tell me everything", "comprehensive", "in-depth"]):
                classification["primary_task"] = "web_search"
                classification["confidence"] = 0.8
                classification["complexity"] = "low"
                return classification

        # 3. Check for research requests (detailed/complex)
        research_indicators = ["research", "explain in detail", "tell me everything", "comprehensive", "in-depth", "analysis"]
        if any(indicator in message_lower for indicator in research_indicators):
            classification["primary_task"] = "research"
            classification["confidence"] = 0.8
            classification["complexity"] = "high"
            return classification

        # 4. Check for coding requests
        coding_indicators = ["code", "programming", "script", "function", "debug", "fix", "write code"]
        if any(indicator in message_lower for indicator in coding_indicators):
            classification["primary_task"] = "coding"
            classification["confidence"] = 0.8
            classification["complexity"] = "medium"
            return classification

        # 5. Check for special requests
        if any(word in message_lower for word in ["personality", "change mode", "switch to"]):
            classification["primary_task"] = "personality"
            classification["confidence"] = 0.9
            return classification

        if any(word in message_lower for word in ["remember", "recall", "memory", "conversation history"]):
            classification["primary_task"] = "memory"
            classification["confidence"] = 0.9
            return classification

        # 6. Default to general conversation
        classification["primary_task"] = "general"
        classification["confidence"] = 0.5
        classification["complexity"] = "low"

        return classification

class CrewOrchestrator:
    """Orchestrates CrewAI agents and tasks based on user requests."""
    
    def __init__(self):
        self.classifier = TaskClassifier()
        self.active_crews: Dict[str, Crew] = {}

    def _clean_final_response(self, response: str) -> str:
        """Final cleanup of response to remove any remaining unwanted elements."""
        import re

        if not response:
            return response

        # Remove thinking tags and their content (case insensitive, multiline)
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL | re.IGNORECASE)
        response = re.sub(r'<thinking>.*?</thinking>', '', response, flags=re.DOTALL | re.IGNORECASE)
        response = re.sub(r'<thought>.*?</thought>', '', response, flags=re.DOTALL | re.IGNORECASE)
        response = re.sub(r'<internal>.*?</internal>', '', response, flags=re.DOTALL | re.IGNORECASE)
        response = re.sub(r'<reasoning>.*?</reasoning>', '', response, flags=re.DOTALL | re.IGNORECASE)

        # Remove any remaining XML-like tags that might contain thinking
        response = re.sub(r'<[^>]*thinking[^>]*>.*?</[^>]*>', '', response, flags=re.DOTALL | re.IGNORECASE)

        # Remove repeated identical lines (like "35\n35\n35\n34")
        lines = response.split('\n')
        cleaned_lines = []
        prev_line = None
        repeat_count = 0

        for line in lines:
            line = line.strip()
            if line == prev_line:
                repeat_count += 1
                # Skip if we've seen this line more than twice in a row
                if repeat_count <= 1:
                    cleaned_lines.append(line)
            else:
                cleaned_lines.append(line)
                prev_line = line
                repeat_count = 0

        response = '\n'.join(cleaned_lines)

        # Remove lines that look like thinking/counting (repeated numbers)
        lines = response.split('\n')
        final_lines = []

        for line in lines:
            line = line.strip()
            # Skip lines that are just numbers repeated or look like counting
            if re.match(r'^\d+$', line):
                # Check if this number appears multiple times in the response
                number_count = response.count(line)
                if number_count > 2:  # If the same number appears more than twice, it's likely thinking
                    continue
            final_lines.append(line)

        response = '\n'.join(final_lines)

        # For simple questions asking for numbers, extract just the final answer
        if any(word in response.lower() for word in ["how many", "count", "letters", "characters"]):
            # Look for the last number in the response
            numbers = re.findall(r'\b\d+\b', response)
            if numbers:
                # If we have multiple numbers and they look like thinking, return just the last one
                if len(numbers) > 1:
                    return numbers[-1]

        # Clean up excessive whitespace
        response = re.sub(r'\n\s*\n\s*\n+', '\n\n', response)  # Multiple newlines to double
        response = re.sub(r'^\s+|\s+$', '', response)  # Trim whitespace

        # Remove empty lines
        response = re.sub(r'\n\s*\n', '\n', response)

        # If response is empty after cleaning, provide a fallback
        if not response.strip():
            return "I'm here to help! What would you like to know or discuss?"

        return response
    
    async def process_request(
        self,
        user_message: str,
        user_id: str,
        channel_id: str = None
    ) -> str:
        """Process a user request and return the appropriate response."""
        try:
            # Classify the request
            classification = self.classifier.classify_request(user_message)
            logger.info(f"Task classification: {classification}")

            # Get user's personality context
            personality = personality_manager.get_user_personality(user_id)
            system_message = personality.system_message
            logger.info(f"Using personality '{personality.name}' for user {user_id}")
            logger.info(f"System message preview: {system_message[:100]}...")
            
            # Handle special cases first
            if classification["primary_task"] == "personality":
                return await self._handle_personality_change(user_message, user_id)
            
            if classification["primary_task"] == "memory":
                return await self._handle_memory_request(user_message, user_id)
            
            # Get intelligent conversation context
            context = memory_manager.get_intelligent_context(user_id, user_message)
            
            # Route to appropriate handler
            if classification["primary_task"] == "image":
                response = await self._handle_image_request(
                    user_message, user_id, context, system_message
                )
            elif classification["primary_task"] == "web_search":
                response = await self._handle_web_search_request(
                    user_message, user_id, context, system_message
                )
            elif classification["primary_task"] == "research":
                response = await self._handle_research_request(
                    user_message, user_id, context, system_message
                )
            elif classification["primary_task"] == "coding":
                response = await self._handle_coding_request(
                    user_message, user_id, context, system_message
                )
            else:
                response = await self._handle_general_request(
                    user_message, user_id, context, system_message
                )
            
            # Clean the response one more time to ensure no thinking tags remain
            logger.info(f"Response before final cleaning: {response[:200]}...")
            response = self._clean_final_response(response)
            logger.info(f"Response after final cleaning: {response[:200]}...")

            # Update memory with conversation messages
            memory_manager.add_conversation_message(user_id, user_message, "user")
            memory_manager.add_conversation_message(user_id, response, "assistant")

            # Add to short-term memory for context
            memory_manager.add_short_term_memory(
                user_id,
                f"User: {user_message}\nAssistant: {response}",
                "conversation",
                importance=0.6
            )

            # Store important information in long-term memory if needed
            if classification["complexity"] == "high" or len(response) > 200:
                await memory_manager.add_long_term_memory(
                    user_id,
                    f"Q: {user_message}\nA: {response[:500]}...",
                    memory_type="conversation",
                    importance=0.8,
                    metadata={"task": classification["primary_task"], "complexity": classification["complexity"]}
                )

            # Extract and store user preferences or facts
            await self._extract_and_store_user_info(user_message, response, user_id)

            return response
            
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            return "I apologize, but I encountered an error while processing your request. Please try again."

    async def _handle_image_request(
        self,
        user_message: str,
        user_id: str,
        context: Dict,
        system_message: str
    ) -> str:
        """Handle image search requests - focused purely on finding images."""
        try:
            logger.info(f"Image search request: {user_message}")

            # Use the dedicated web search executor for images
            result = await web_search_executor.execute_search_task(user_message, user_id, "image")

            logger.info("Image search completed - returning direct results")
            return result

        except Exception as e:
            logger.error(f"Error in image request: {e}")
            # Simple fallback
            return f"I encountered an error searching for images. Please try searching manually for images of: {user_message}"

    async def _handle_web_search_request(
        self,
        user_message: str,
        user_id: str,
        context: Dict,
        system_message: str
    ) -> str:
        """Handle simple web search requests - focused on current info and basic searches."""
        try:
            logger.info(f"Simple web search request: {user_message}")

            # Determine search type
            search_type = "general"
            if any(term in user_message.lower() for term in ["stock price", "price of", "current price"]):
                search_type = "current"
            elif any(term in user_message.lower() for term in ["news", "latest", "recent", "today"]):
                search_type = "current"

            # Use the dedicated web search executor
            result = await web_search_executor.execute_search_task(user_message, user_id, search_type)

            logger.info("Web search completed - returning direct results")
            return result

        except Exception as e:
            logger.error(f"Error in web search request: {e}")
            return f"I encountered an issue while searching. Please try again: {str(e)}"

    async def _handle_research_request(
        self,
        user_message: str,
        user_id: str,
        context: Dict,
        system_message: str
    ) -> str:
        """Handle research-related requests."""
        try:
            # Use the research executor
            research_result = await research_executor.execute_research_task(user_message, user_id)

            # Enhance with LLM response using personality and context
            context_info = self._format_context_for_llm(context)
            enhanced_response = await llm_manager.generate_response(
                user_message=f"Based on this research: {research_result}\n\nUser question: {user_message}",
                system_message=system_message,
                conversation_history=context.get("conversation_history", [])[-3:],  # Last 3 messages
                max_tokens=config.research_max_tokens  # Configurable limit for research responses
            )

            return enhanced_response or research_result

        except Exception as e:
            logger.error(f"Error in research request: {e}")
            return f"I encountered an issue while researching your question. Here's what I found: {str(e)}"
    
    async def _handle_coding_request(
        self,
        user_message: str,
        user_id: str,
        context: Dict,
        system_message: str
    ) -> str:
        """Handle coding-related requests."""
        try:
            # Use the coding executor
            coding_result = await coding_executor.execute_coding_task(user_message, user_id)
            
            # Enhance with LLM response using personality and context
            context_info = self._format_context_for_llm(context)
            enhanced_response = await llm_manager.generate_response(
                user_message=f"Based on this coding analysis: {coding_result}\n\nUser request: {user_message}",
                system_message=system_message,
                conversation_history=context.get("conversation_history", [])[-3:],
                max_tokens=config.coding_max_tokens  # Configurable limit for coding responses
            )
            
            return enhanced_response or coding_result
            
        except Exception as e:
            logger.error(f"Error in coding request: {e}")
            return f"I encountered an issue while working on your coding request: {str(e)}"
    
    async def _handle_general_request(
        self,
        user_message: str,
        user_id: str,
        context: Dict,
        system_message: str
    ) -> str:
        """Handle general conversation and questions."""
        try:
            # Prepare context for LLM with intelligent memory
            context_info = self._format_context_for_llm(context)

            # Generate response using LLM with personality and context
            response = await llm_manager.generate_response(
                user_message=f"{context_info}{user_message}",
                system_message=system_message,
                conversation_history=context.get("conversation_history", [])[-5:],  # Last 5 messages
                max_tokens=config.general_max_tokens  # Configurable limit for general responses
            )
            
            return response or "I'm here to help! Could you please rephrase your question?"
            
        except Exception as e:
            logger.error(f"Error in general request: {e}")
            return "I'm having trouble processing your request right now. Please try again."
    
    async def _handle_personality_change(self, user_message: str, user_id: str) -> str:
        """Handle personality change requests."""
        message_lower = user_message.lower()
        
        # Extract personality name from message
        personalities = personality_manager.list_personalities()
        
        for personality_name in personalities.keys():
            if personality_name in message_lower:
                success = personality_manager.set_user_personality(user_id, personality_name)
                if success:
                    personality = personality_manager.get_personality(personality_name)
                    return f"I've switched to {personality_name} personality! {personality.description}"
                else:
                    return f"I couldn't switch to {personality_name} personality. Please try again."
        
        # If no specific personality mentioned, list available ones
        personality_list = "\n".join([f"- **{name}**: {desc}" for name, desc in personalities.items()])
        return f"Available personalities:\n\n{personality_list}\n\nSay something like 'switch to casual personality' to change modes."
    
    async def _handle_memory_request(self, user_message: str, user_id: str) -> str:
        """Handle memory-related requests."""
        if "clear" in user_message.lower() or "reset" in user_message.lower():
            if "conversation" in user_message.lower():
                memory_manager.clear_conversation_history(user_id)
                return "I've cleared our conversation history. We can start fresh!"
            else:
                memory_manager.clear_short_term_memory(user_id)
                memory_manager.clear_conversation_history(user_id)
                return "I've cleared all our conversation memory. We can start completely fresh!"

        if "stats" in user_message.lower():
            stats = memory_manager.get_memory_stats(user_id)
            return f"""Memory Statistics for you:
- Conversation messages: {stats['conversation_messages']}
- Short-term memories: {stats['short_term_memories']}
- Long-term memories: {stats.get('long_term_memories', 'unknown')}
- Preferences stored: {stats['user_preferences']}
- Facts about you: {stats['user_facts']}"""

        # Retrieve and summarize memory
        context = memory_manager.get_intelligent_context(user_id, user_message)

        if context.get("conversation_history"):
            recent_count = len(context["conversation_history"])
            relevant_count = len(context.get("relevant_memories", []))
            return f"I remember our recent conversation ({recent_count} messages) and have {relevant_count} relevant memories about our past discussions. What would you like to know?"
        else:
            return "We haven't had any previous conversations that I can recall. What would you like to talk about?"

    def _format_context_for_llm(self, context: Dict) -> str:
        """Format intelligent context for LLM consumption."""
        context_parts = []

        # Add user preferences if available
        if context.get("user_preferences"):
            prefs = ", ".join([f"{k}: {v}" for k, v in context["user_preferences"].items()])
            context_parts.append(f"User preferences: {prefs}")

        # Add relevant memories if available
        if context.get("relevant_memories"):
            memories = []
            for memory in context["relevant_memories"][:3]:  # Top 3 most relevant
                memory_type = memory.get("memory_type", "memory")
                content = memory.get("content", "")[:100]  # Truncate long memories
                memories.append(f"[{memory_type}] {content}")

            if memories:
                context_parts.append(f"Relevant memories: {'; '.join(memories)}")

        # Add context summary
        if context.get("context_summary"):
            context_parts.append(f"Context: {context['context_summary']}")

        # Add detected intent
        if context.get("intent") and context["intent"] != "general":
            context_parts.append(f"Detected intent: {context['intent']}")

        if context_parts:
            return "\n".join(context_parts) + "\n\n"
        return ""

    async def _extract_and_store_user_info(self, user_message: str, response: str, user_id: str):
        """Extract and store user preferences and facts from conversation."""
        try:
            message_lower = user_message.lower()

            # Extract preferences
            preference_indicators = [
                "i prefer", "i like", "i love", "i hate", "i dislike",
                "my favorite", "i usually", "i always", "i never"
            ]

            for indicator in preference_indicators:
                if indicator in message_lower:
                    # Simple extraction - can be enhanced with NLP
                    preference_text = user_message[message_lower.find(indicator):]
                    memory_manager.add_user_fact(user_id, preference_text, importance=0.8)
                    break

            # Extract facts about user
            fact_indicators = [
                "i am", "i'm", "i work", "i live", "my name is",
                "i study", "i'm from", "i have", "i own"
            ]

            for indicator in fact_indicators:
                if indicator in message_lower:
                    fact_text = user_message[message_lower.find(indicator):]
                    memory_manager.add_user_fact(user_id, fact_text, importance=0.7)
                    break

            # Store important technical information
            if any(word in message_lower for word in ["programming", "code", "language", "framework", "technology"]):
                await memory_manager.add_long_term_memory(
                    user_id,
                    f"Technical discussion: {user_message}",
                    "technical",
                    importance=0.7
                )

        except Exception as e:
            logger.error(f"Error extracting user info: {e}")

# Global orchestrator instance
crew_orchestrator = CrewOrchestrator()
