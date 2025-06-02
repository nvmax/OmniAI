"""
Vector database tool for memory retrieval and context search.
Integrates with the memory manager for intelligent information retrieval.
"""

import logging
from typing import List, Dict, Any, Optional
from crewai.tools.base_tool import BaseTool

try:
    from ..memory_manager import memory_manager
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from memory_manager import memory_manager

logger = logging.getLogger(__name__)

class VectorDBRetrievalTool(BaseTool):
    """Tool for retrieving relevant information from the vector database."""
    
    name: str = "memory_search"
    description: str = (
        "Search through conversation history and stored knowledge to find relevant information. "
        "Provide search terms or questions to retrieve related context from past conversations."
    )
    
    def __init__(self):
        super().__init__()
        self._memory_manager = memory_manager
    
    def _run(self, query: str, user_id: str = "default") -> str:
        """Search for relevant information in memory."""
        try:
            # Get intelligent conversation context
            context = self._memory_manager.get_intelligent_context(user_id, query)

            result_parts = []

            # Add conversation history
            if context.get("conversation_history"):
                recent_messages = context["conversation_history"][-5:]  # Last 5 messages
                result_parts.append("Recent conversation context:")
                for msg in recent_messages:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")[:200]  # Truncate long messages
                    result_parts.append(f"- {role}: {content}")
                result_parts.append("")

            # Add relevant memories
            if context.get("relevant_memories"):
                result_parts.append("Relevant past information:")
                for memory in context["relevant_memories"]:
                    content = memory["content"][:300]  # Truncate long content
                    memory_type = memory.get("memory_type", "unknown")
                    relevance = memory.get("relevance_score", 0)
                    timestamp = memory.get("metadata", {}).get("timestamp", "Unknown time")
                    result_parts.append(f"- [{memory_type}] {content} (relevance: {relevance:.2f}, from {timestamp})")
                result_parts.append("")

            # Add user preferences
            if context.get("user_preferences"):
                result_parts.append("User preferences:")
                for key, value in context["user_preferences"].items():
                    result_parts.append(f"- {key}: {value}")
                result_parts.append("")

            # Add context summary
            if context.get("context_summary"):
                result_parts.append(f"Summary: {context['context_summary']}")

            # Add detected intent
            if context.get("intent") and context["intent"] != "general":
                result_parts.append(f"Detected intent: {context['intent']}")

            if not result_parts:
                return f"No relevant information found for query: '{query}'"

            return "\n".join(result_parts)

        except Exception as e:
            logger.error(f"Error searching memory: {e}")
            return f"Error searching memory: {str(e)}"

class MemoryStorageTool(BaseTool):
    """Tool for storing important information in long-term memory."""
    
    name: str = "memory_storage"
    description: str = (
        "Store important information in long-term memory for future reference. "
        "Use this to save key facts, preferences, or important conversation points."
    )
    
    def __init__(self):
        super().__init__()
        self._memory_manager = memory_manager
    
    def _run(self, content: str, user_id: str = "default", category: str = "general") -> str:
        """Store information in long-term memory."""
        try:
            import asyncio

            # Prepare metadata
            metadata = {
                "category": category,
                "source": "agent_storage"
            }

            # Store in long-term memory (async-safe)
            try:
                # Check if we're in an async context
                asyncio.get_running_loop()
                # In async context, use add_user_fact instead for immediate storage
                self._memory_manager.add_user_fact(user_id, content, importance=0.7)
                return f"Successfully stored information in memory: '{content[:100]}...'"
            except RuntimeError:
                # Not in async context, safe to create new loop
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(
                    self._memory_manager.add_long_term_memory(
                        user_id, content, memory_type=category, importance=0.7, metadata=metadata
                    )
                )
                loop.close()
                return f"Successfully stored information in long-term memory: '{content[:100]}...'"

        except Exception as e:
            logger.error(f"Error storing memory: {e}")
            return f"Error storing information: {str(e)}"

class ContextAnalysisTool(BaseTool):
    """Tool for analyzing conversation context and extracting key information."""
    
    name: str = "context_analysis"
    description: str = (
        "Analyze the current conversation context to extract key topics, "
        "user preferences, and important information that should be remembered."
    )
    
    def __init__(self):
        super().__init__()
        self._memory_manager = memory_manager
    
    def _run(self, user_id: str = "default") -> str:
        """Analyze conversation context for a user."""
        try:
            # Get memory statistics
            stats = self._memory_manager.get_memory_stats(user_id)

            # Get recent conversation history
            conversation_history = self._memory_manager.get_conversation_history(user_id, limit=10)
            short_term = self._memory_manager.get_short_term_memory(user_id, limit=5)

            if not conversation_history and not short_term:
                return "No conversation history available for analysis."

            # Extract key information
            analysis_parts = []

            analysis_parts.append(f"Conversation Analysis for User {user_id}:")
            analysis_parts.append(f"- Conversation messages: {stats['conversation_messages']}")
            analysis_parts.append(f"- Short-term memories: {stats['short_term_memories']}")
            analysis_parts.append(f"- Long-term memories: {stats.get('long_term_memories', 'unknown')}")
            analysis_parts.append(f"- User preferences: {stats['user_preferences']}")
            analysis_parts.append(f"- User facts: {stats['user_facts']}")

            # Analyze conversation history
            if conversation_history:
                user_messages = [msg for msg in conversation_history if msg.get("role") == "user"]
                bot_messages = [msg for msg in conversation_history if msg.get("role") == "assistant"]

                analysis_parts.append(f"- Recent user messages: {len(user_messages)}")
                analysis_parts.append(f"- Recent bot responses: {len(bot_messages)}")

                # Extract topics from conversation
                all_content = " ".join([msg.get("content", "") for msg in conversation_history])
                words = all_content.lower().split()

                # Common technical terms that might indicate topics
                topic_keywords = {
                    "programming": ["code", "python", "javascript", "programming", "function", "class", "coding"],
                    "discord": ["discord", "bot", "server", "channel", "message"],
                    "ai": ["ai", "artificial", "intelligence", "machine", "learning", "model"],
                    "web": ["web", "website", "html", "css", "browser", "http"],
                    "data": ["data", "database", "sql", "json", "api"],
                    "memory": ["memory", "remember", "recall", "forget", "preference"]
                }

                detected_topics = []
                for topic, keywords in topic_keywords.items():
                    if any(keyword in words for keyword in keywords):
                        detected_topics.append(topic)

                if detected_topics:
                    analysis_parts.append(f"- Detected topics: {', '.join(detected_topics)}")

                # Recent activity summary
                if user_messages:
                    recent_user_msg = user_messages[-1]["content"][:100]
                    analysis_parts.append(f"- Latest user message: {recent_user_msg}...")

            # Analyze short-term memory patterns
            if short_term:
                memory_types = [entry.memory_type for entry in short_term]
                type_counts = {}
                for mem_type in memory_types:
                    type_counts[mem_type] = type_counts.get(mem_type, 0) + 1

                if type_counts:
                    type_summary = ", ".join([f"{t}: {c}" for t, c in type_counts.items()])
                    analysis_parts.append(f"- Memory types: {type_summary}")

            return "\n".join(analysis_parts)

        except Exception as e:
            logger.error(f"Error analyzing context: {e}")
            return f"Error analyzing context: {str(e)}"
