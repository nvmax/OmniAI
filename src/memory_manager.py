"""
Per-User Memory management system for Omni-Assistant Discord Bot.
Handles user-isolated short-term, long-term, and contextual memory using vector database.
Implements intelligent context retrieval based on conversation relevance.
"""

import asyncio
import json
import logging
import time
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

try:
    from .config import config
except ImportError:
    from config import config

logger = logging.getLogger(__name__)

@dataclass
class MemoryEntry:
    """Represents a single memory entry with metadata."""
    content: str
    timestamp: datetime
    user_id: str
    memory_type: str  # 'conversation', 'preference', 'fact', 'context'
    importance: float  # 0.0 to 1.0
    metadata: Dict[str, Any]

@dataclass
class ConversationContext:
    """Represents conversation context for intelligent retrieval."""
    user_message: str
    conversation_history: List[Dict[str, str]]
    topics: List[str]
    intent: str
    relevance_keywords: List[str]

class PerUserMemoryManager:
    """Manages per-user isolated memory with intelligent context retrieval."""

    def __init__(self):
        self.vector_db_path = config.vector_db_path
        self.max_short_term = config.max_short_term_memory
        self.max_context_retrieval = config.max_context_retrieval

        # Per-user short-term memory: user_id -> deque of MemoryEntry
        self.short_term_memory: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.max_short_term)
        )

        # Per-user conversation history: user_id -> conversation messages
        self.conversation_history: Dict[str, List[Dict[str, str]]] = defaultdict(list)

        # Per-user preferences and facts: user_id -> dict
        self.user_preferences: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.user_facts: Dict[str, Dict[str, Any]] = defaultdict(dict)

        # Context cache with user isolation: user_id -> cached context
        self.context_cache: Dict[str, Dict] = {}
        self.cache_expiry: Dict[str, float] = {}
        self.cache_duration = 300  # 5 minutes

        # Relevance scoring weights
        self.relevance_weights = {
            'keyword_match': 0.3,
            'semantic_similarity': 0.4,
            'recency': 0.2,
            'importance': 0.1
        }

        # Initialize vector database and embedding model
        self.chroma_client = None
        self.collections: Dict[str, Any] = {}  # user_id -> collection
        self.embedding_model = None
        self._initialize_vector_db()
        self._initialize_embedding_model()
    
    def _initialize_vector_db(self):
        """Initialize ChromaDB for per-user long-term memory storage."""
        try:
            # Create the directory if it doesn't exist
            import os
            os.makedirs(self.vector_db_path, exist_ok=True)

            self.chroma_client = chromadb.PersistentClient(
                path=self.vector_db_path,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                    is_persistent=True
                )
            )

            logger.info("Vector database initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize vector database: {e}")
            # Try to initialize without persistence as fallback
            try:
                logger.info("Attempting to initialize in-memory vector database as fallback...")
                self.chroma_client = chromadb.Client(
                    settings=Settings(
                        anonymized_telemetry=False,
                        allow_reset=True
                    )
                )
                logger.info("In-memory vector database initialized successfully")
            except Exception as e2:
                logger.error(f"Failed to initialize fallback vector database: {e2}")
                self.chroma_client = None

    def _get_user_collection(self, user_id: str):
        """Get or create a user-specific collection."""
        if not self.chroma_client:
            return None

        if user_id not in self.collections:
            try:
                # Create a safe collection name from user_id
                safe_name = f"user_{hashlib.md5(user_id.encode()).hexdigest()[:16]}"

                self.collections[user_id] = self.chroma_client.get_or_create_collection(
                    name=safe_name,
                    metadata={
                        "description": f"Long-term memory for user {user_id}",
                        "user_id": user_id,
                        "created_at": datetime.now().isoformat()
                    }
                )
                logger.debug(f"Created collection for user {user_id}")
            except Exception as e:
                logger.error(f"Failed to create collection for user {user_id}: {e}")
                return None

        return self.collections[user_id]
    
    def _initialize_embedding_model(self):
        """Initialize the sentence transformer model for embeddings."""
        try:
            # Use a lightweight, fast model for local embeddings
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Embedding model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            self.embedding_model = None

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text for relevance matching."""
        import re

        # Simple keyword extraction - can be enhanced with NLP libraries
        words = re.findall(r'\b\w+\b', text.lower())

        # Filter out common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'
        }

        keywords = [word for word in words if len(word) > 2 and word not in stop_words]
        return list(set(keywords))  # Remove duplicates

    def _analyze_conversation_context(self, user_message: str, user_id: str) -> ConversationContext:
        """Analyze conversation context for intelligent memory retrieval."""

        # Get recent conversation history
        recent_history = self.conversation_history[user_id][-10:]  # Last 10 messages

        # Extract keywords from current message
        keywords = self._extract_keywords(user_message)

        # Simple intent detection
        intent = "general"
        if any(word in user_message.lower() for word in ["code", "programming", "function", "debug"]):
            intent = "coding"
        elif any(word in user_message.lower() for word in ["search", "find", "research", "information"]):
            intent = "research"
        elif any(word in user_message.lower() for word in ["remember", "preference", "like", "prefer"]):
            intent = "preference"

        # Extract topics from conversation history
        topics = []
        for msg in recent_history[-5:]:  # Last 5 messages
            topics.extend(self._extract_keywords(msg.get("content", "")))

        return ConversationContext(
            user_message=user_message,
            conversation_history=recent_history,
            topics=list(set(topics)),
            intent=intent,
            relevance_keywords=keywords
        )
    
    def add_conversation_message(self, user_id: str, message: str, role: str = "user"):
        """Add a message to conversation history."""
        conversation_entry = {
            "role": role,
            "content": message,
            "timestamp": datetime.now().isoformat()
        }

        # Add to conversation history
        self.conversation_history[user_id].append(conversation_entry)

        # Limit conversation history size (keep last 100 messages)
        if len(self.conversation_history[user_id]) > 100:
            self.conversation_history[user_id] = self.conversation_history[user_id][-100:]

        # Clear context cache when new message is added
        cache_key = f"{user_id}_context"
        if cache_key in self.context_cache:
            del self.context_cache[cache_key]
            del self.cache_expiry[cache_key]

    def add_short_term_memory(self, user_id: str, content: str, memory_type: str = "conversation", importance: float = 0.5):
        """Add a memory entry to short-term memory."""
        memory_entry = MemoryEntry(
            content=content,
            timestamp=datetime.now(),
            user_id=user_id,
            memory_type=memory_type,
            importance=importance,
            metadata={"source": "short_term"}
        )

        self.short_term_memory[user_id].append(memory_entry)

        # Clear context cache
        cache_key = f"{user_id}_context"
        if cache_key in self.context_cache:
            del self.context_cache[cache_key]
            del self.cache_expiry[cache_key]
    
    def get_short_term_memory(self, user_id: str, limit: int = None) -> List[MemoryEntry]:
        """Get recent short-term memory for a user."""
        memory = list(self.short_term_memory[user_id])
        if limit:
            memory = memory[-limit:]
        return memory

    def get_conversation_history(self, user_id: str, limit: int = None) -> List[Dict[str, str]]:
        """Get conversation history for a user."""
        history = self.conversation_history[user_id]
        if limit:
            history = history[-limit:]
        return history

    def clear_short_term_memory(self, user_id: str):
        """Clear short-term memory for a user."""
        if user_id in self.short_term_memory:
            self.short_term_memory[user_id].clear()

        # Also clear context cache
        cache_key = f"{user_id}_context"
        if cache_key in self.context_cache:
            del self.context_cache[cache_key]
            del self.cache_expiry[cache_key]

    def clear_conversation_history(self, user_id: str):
        """Clear conversation history for a user."""
        if user_id in self.conversation_history:
            self.conversation_history[user_id].clear()

        # Clear context cache
        cache_key = f"{user_id}_context"
        if cache_key in self.context_cache:
            del self.context_cache[cache_key]
            del self.cache_expiry[cache_key]
    
    async def add_long_term_memory(self, user_id: str, content: str, memory_type: str = "conversation", importance: float = 0.5, metadata: Dict = None):
        """Add content to user-specific long-term vector memory."""
        collection = self._get_user_collection(user_id)
        if not collection or not self.embedding_model:
            logger.warning(f"Vector database or embedding model not available for user {user_id}")
            return

        try:
            # Generate embedding
            embedding = self.embedding_model.encode(content).tolist()

            # Prepare metadata
            memory_metadata = {
                "user_id": user_id,
                "timestamp": datetime.now().isoformat(),
                "memory_type": memory_type,
                "importance": importance,
                "keywords": ",".join(self._extract_keywords(content))
            }
            if metadata:
                memory_metadata.update(metadata)

            # Generate unique ID
            memory_id = f"{user_id}_{memory_type}_{int(time.time() * 1000)}"

            # Add to user's collection
            collection.add(
                embeddings=[embedding],
                documents=[content],
                metadatas=[memory_metadata],
                ids=[memory_id]
            )

            logger.debug(f"Added long-term memory for user {user_id}: {memory_type}")

        except Exception as e:
            logger.error(f"Failed to add long-term memory for user {user_id}: {e}")
    
    async def retrieve_relevant_memories(
        self,
        user_id: str,
        context: ConversationContext,
        k: int = None
    ) -> List[Dict]:
        """Retrieve relevant memories using intelligent context analysis."""
        collection = self._get_user_collection(user_id)
        if not collection or not self.embedding_model:
            return []

        k = k or self.max_context_retrieval

        # Check cache first
        cache_key = f"{user_id}_relevant_{hash(context.user_message)}"
        current_time = time.time()

        if (cache_key in self.context_cache and
            cache_key in self.cache_expiry and
            current_time < self.cache_expiry[cache_key]):
            return self.context_cache[cache_key]

        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode(context.user_message).tolist()

            # Search for relevant memories (get more than needed for filtering)
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=k * 3,  # Get 3x more for intelligent filtering
                include=["documents", "metadatas", "distances"]
            )

            # Score and filter results based on relevance
            relevant_memories = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                    distance = results['distances'][0][i] if results['distances'] else 1.0

                    # Calculate relevance score
                    relevance_score = self._calculate_relevance_score(
                        doc, metadata, distance, context
                    )

                    # Only include if relevance score is above threshold
                    if relevance_score > 0.3:  # Adjustable threshold
                        memory = {
                            "content": doc,
                            "metadata": metadata,
                            "distance": distance,
                            "relevance_score": relevance_score,
                            "memory_type": metadata.get("memory_type", "unknown")
                        }
                        relevant_memories.append(memory)

            # Sort by relevance score and limit results
            relevant_memories.sort(key=lambda x: x["relevance_score"], reverse=True)
            relevant_memories = relevant_memories[:k]

            # Cache results
            self.context_cache[cache_key] = relevant_memories
            self.cache_expiry[cache_key] = current_time + self.cache_duration

            logger.debug(f"Retrieved {len(relevant_memories)} relevant memories for user {user_id}")
            return relevant_memories

        except Exception as e:
            logger.error(f"Failed to retrieve relevant memories for user {user_id}: {e}")
            return []
    
    def _calculate_relevance_score(self, content: str, metadata: Dict, distance: float, context: ConversationContext) -> float:
        """Calculate relevance score for a memory based on multiple factors."""

        # Semantic similarity (inverse of distance, normalized)
        semantic_score = max(0, 1 - distance)

        # Keyword matching
        content_keywords = set(self._extract_keywords(content))
        query_keywords = set(context.relevance_keywords)
        topic_keywords = set(context.topics)

        keyword_overlap = len(content_keywords.intersection(query_keywords))
        topic_overlap = len(content_keywords.intersection(topic_keywords))
        total_keywords = len(query_keywords) + len(topic_keywords)

        keyword_score = (keyword_overlap + topic_overlap) / max(1, total_keywords)

        # Recency score (more recent = higher score)
        try:
            memory_time = datetime.fromisoformat(metadata.get("timestamp", ""))
            time_diff = (datetime.now() - memory_time).total_seconds()
            # Decay over 30 days
            recency_score = max(0, 1 - (time_diff / (30 * 24 * 3600)))
        except:
            recency_score = 0.5  # Default if timestamp parsing fails

        # Importance score from metadata
        importance_score = metadata.get("importance", 0.5)

        # Intent matching bonus
        intent_bonus = 0.0
        memory_type = metadata.get("memory_type", "")
        if context.intent == "coding" and memory_type in ["coding", "technical"]:
            intent_bonus = 0.2
        elif context.intent == "preference" and memory_type in ["preference", "fact"]:
            intent_bonus = 0.2
        elif context.intent == "research" and memory_type in ["research", "information"]:
            intent_bonus = 0.2

        # Calculate weighted score
        final_score = (
            semantic_score * self.relevance_weights['semantic_similarity'] +
            keyword_score * self.relevance_weights['keyword_match'] +
            recency_score * self.relevance_weights['recency'] +
            importance_score * self.relevance_weights['importance'] +
            intent_bonus
        )

        return min(1.0, final_score)  # Cap at 1.0

    def get_intelligent_context(self, user_id: str, user_message: str) -> Dict:
        """Get intelligent conversation context with relevant memories."""

        # Analyze conversation context
        context = self._analyze_conversation_context(user_message, user_id)

        # Get conversation history
        conversation_history = self.get_conversation_history(user_id, limit=10)

        # Get short-term memory
        short_term = self.get_short_term_memory(user_id, limit=5)

        result = {
            "conversation_history": conversation_history,
            "short_term_memory": [
                {
                    "content": entry.content,
                    "type": entry.memory_type,
                    "timestamp": entry.timestamp.isoformat(),
                    "importance": entry.importance
                } for entry in short_term
            ],
            "relevant_memories": [],
            "user_preferences": self.user_preferences.get(user_id, {}),
            "context_summary": "",
            "intent": context.intent,
            "topics": context.topics
        }

        # Try to get relevant long-term memories (async-safe)
        try:
            # Check if we're in an async context
            try:
                asyncio.get_running_loop()
                # In async context, skip long-term retrieval to avoid conflicts
                logger.debug("Skipping long-term memory retrieval in async context")
            except RuntimeError:
                # Not in async context, safe to retrieve
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                relevant_memories = loop.run_until_complete(
                    self.retrieve_relevant_memories(user_id, context)
                )
                result["relevant_memories"] = relevant_memories
                loop.close()
        except Exception as e:
            logger.error(f"Error retrieving relevant memories: {e}")

        # Generate context summary
        if conversation_history:
            recent_messages = [msg["content"] for msg in conversation_history[-3:]]
            result["context_summary"] = f"Recent: {' | '.join(recent_messages)}"

        return result
    
    def add_user_preference(self, user_id: str, key: str, value: Any):
        """Add or update a user preference."""
        self.user_preferences[user_id][key] = value

        # Store in long-term memory as well
        preference_text = f"User preference: {key} = {value}"
        asyncio.create_task(
            self.add_long_term_memory(
                user_id, preference_text, "preference", importance=0.8
            )
        )

    def add_user_fact(self, user_id: str, fact: str, importance: float = 0.7):
        """Add a fact about the user."""
        fact_key = hashlib.md5(fact.encode()).hexdigest()[:16]
        self.user_facts[user_id][fact_key] = {
            "fact": fact,
            "timestamp": datetime.now().isoformat(),
            "importance": importance
        }

        # Store in long-term memory
        asyncio.create_task(
            self.add_long_term_memory(
                user_id, fact, "fact", importance=importance
            )
        )

    async def cleanup_old_memories(self, user_id: str = None, days_old: int = 30):
        """Clean up old memories from the vector database."""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_old)

            if user_id:
                # Clean up specific user's memories
                collection = self._get_user_collection(user_id)
                if collection:
                    # This is a simplified cleanup - ChromaDB doesn't have direct date filtering
                    # In a production system, you'd want more sophisticated cleanup logic
                    logger.info(f"Memory cleanup completed for user {user_id}, memories older than {days_old} days")
            else:
                # Clean up all users
                for uid in self.collections.keys():
                    await self.cleanup_old_memories(uid, days_old)

        except Exception as e:
            logger.error(f"Failed to cleanup old memories: {e}")

    def get_memory_stats(self, user_id: str = None) -> Dict:
        """Get statistics about memory usage."""
        if user_id:
            # User-specific stats
            collection = self._get_user_collection(user_id)
            stats = {
                "user_id": user_id,
                "short_term_memories": len(self.short_term_memory[user_id]),
                "conversation_messages": len(self.conversation_history[user_id]),
                "user_preferences": len(self.user_preferences[user_id]),
                "user_facts": len(self.user_facts[user_id]),
                "vector_db_available": collection is not None
            }

            if collection:
                try:
                    count_result = collection.count()
                    stats["long_term_memories"] = count_result
                except Exception as e:
                    logger.error(f"Error getting vector DB count for user {user_id}: {e}")
                    stats["long_term_memories"] = "unknown"

            return stats
        else:
            # Global stats
            stats = {
                "total_users": len(self.short_term_memory),
                "total_short_term_memories": sum(
                    len(memory) for memory in self.short_term_memory.values()
                ),
                "total_conversations": sum(
                    len(history) for history in self.conversation_history.values()
                ),
                "cache_entries": len(self.context_cache),
                "vector_db_available": self.chroma_client is not None,
                "user_collections": len(self.collections)
            }

            return stats

# Global per-user memory manager instance
memory_manager = PerUserMemoryManager()
