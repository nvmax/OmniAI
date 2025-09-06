"""
Semantic memory system for general chatbot memory without hardcoded patterns.
Uses relevance scoring and semantic similarity for memory retrieval.
"""

import logging
from typing import List, Dict, Any, Tuple
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)

class SemanticMemoryHandler:
    """Handles memory requests using semantic similarity and relevance scoring."""
    
    def __init__(self, memory_manager):
        self.memory_manager = memory_manager
    
    async def handle_memory_request(self, user_message: str, user_id: str) -> str:
        """Handle memory requests using pure semantic similarity without hardcoded patterns."""
        try:
            logger.info(f"Processing semantic memory request: {user_message}")
            
            # Get all available memories
            conversation_history = self.memory_manager.get_conversation_history(user_id, limit=100)
            
            # Score all memories by relevance
            scored_memories = []
            
            # 1. Score conversation history
            for msg in conversation_history:
                if msg.get('role') == 'user':
                    content = msg.get('content', '')
                    if len(content.strip()) > 3:  # Skip very short messages
                        score = self._calculate_relevance_score(user_message, content)
                        if score > 0.1:  # Only include somewhat relevant memories
                            scored_memories.append({
                                'content': content,
                                'score': score,
                                'type': 'conversation'
                            })
            
            # 2. Get semantic memories if available
            try:
                from memory_manager import ConversationContext
                context = ConversationContext(
                    user_message=user_message,
                    conversation_history=[],
                    topics=user_message.split(),
                    intent="memory_search",
                    relevance_keywords=user_message.split()
                )
                semantic_memories = await self.memory_manager.retrieve_relevant_memories(user_id, context, k=20)
                
                for mem_dict in semantic_memories:
                    content = mem_dict.get('content', '')
                    if content and len(content.strip()) > 3:
                        # Use semantic distance as base score
                        distance = mem_dict.get('distance', 0.5)
                        base_score = max(0, 1.0 - distance)  # Convert distance to similarity
                        
                        # Enhance with text similarity
                        text_score = self._calculate_relevance_score(user_message, content)
                        combined_score = (base_score * 0.6) + (text_score * 0.4)
                        
                        scored_memories.append({
                            'content': content,
                            'score': combined_score,
                            'type': 'semantic'
                        })
                        
            except Exception as e:
                logger.error(f"Error in semantic memory search: {e}")
            
            # 3. Remove duplicates and sort by relevance
            unique_memories = {}
            for mem in scored_memories:
                content = mem['content']
                if content not in unique_memories or mem['score'] > unique_memories[content]['score']:
                    unique_memories[content] = mem
            
            # Sort by relevance score (highest first)
            ranked_memories = sorted(unique_memories.values(), key=lambda x: x['score'], reverse=True)
            
            # 4. Generate response based on top memories
            if ranked_memories:
                top_memories = ranked_memories[:5]  # Top 5 most relevant
                
                # Check if this is a general question
                if self._is_general_question(user_message):
                    return self._format_general_summary(top_memories)
                else:
                    # Try to extract a specific answer
                    answer = self._extract_specific_answer(user_message, top_memories)
                    if answer:
                        return answer
                    
                    # Fallback to showing top relevant memories
                    return self._format_relevant_memories(top_memories[:3])
            
            return "I don't have any relevant information stored yet. Feel free to tell me more!"
                
        except Exception as e:
            logger.error(f"Error handling semantic memory request: {e}")
            return "I'm having trouble accessing my memory right now."
    
    def _calculate_relevance_score(self, question: str, memory_content: str) -> float:
        """Calculate relevance score between question and memory using multiple factors."""
        question_lower = question.lower()
        content_lower = memory_content.lower()

        # 1. Direct text similarity
        similarity = SequenceMatcher(None, question_lower, content_lower).ratio()

        # 2. Enhanced word overlap with semantic matching
        question_words = set(question_lower.split())
        content_words = set(content_lower.split())

        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'what', 'who', 'where', 'when', 'why', 'how', 'do', 'does', 'did', 'my', 'your', 'his', 'her', 'its', 'our', 'their', 'whats', 'i'}
        question_words = question_words - stop_words
        content_words = content_words - stop_words

        # 3. Boost score for any shared meaningful words (general approach)
        shared_meaningful_words = question_words.intersection(content_words)
        word_boost = len(shared_meaningful_words) * 0.2  # Boost for each shared word

        # 4. Direct word overlap score
        if question_words and content_words:
            overlap = len(question_words.intersection(content_words))
            word_score = overlap / max(len(question_words), len(content_words))  # Use max instead of union for better scoring
        else:
            word_score = 0

        # 5. Length penalty (prefer shorter, more direct answers)
        length_penalty = min(1.0, 100 / max(len(memory_content), 1))

        # Combine scores with word boost
        final_score = (similarity * 0.2) + (word_score * 0.4) + (word_boost * 0.3) + (length_penalty * 0.1)

        return min(1.0, final_score)
    
    def _is_general_question(self, question: str) -> bool:
        """Check if this is a general question asking for overview."""
        general_patterns = [
            "what do you know",
            "tell me about",
            "what have we",
            "what information",
            "summary",
            "overview"
        ]
        question_lower = question.lower()
        return any(pattern in question_lower for pattern in general_patterns)
    
    def _extract_specific_answer(self, question: str, memories: List[Dict]) -> str:
        """Extract a specific answer from the most relevant memory using general approach."""
        if not memories:
            return None

        # Filter out memories that are just questions (avoid returning the user's own question)
        answer_memories = []
        for memory in memories:
            content = memory['content'].lower()
            # Skip if this looks like a question rather than a statement
            if not any(q_word in content for q_word in ['what', 'who', 'where', 'when', 'how', '?']):
                answer_memories.append(memory)

        if not answer_memories:
            # If no statement memories found, return the best overall memory but indicate it might not be perfect
            best_memory = memories[0]
            if best_memory['score'] > 0.3:
                return f"I remember you mentioning: {best_memory['content']}"
            return None

        # Get the best answer memory (statement, not question)
        best_memory = answer_memories[0]
        content = best_memory['content']

        # Return the most relevant statement
        if best_memory['score'] > 0.3:
            return f"You told me: {content}"

        return None
    
    def _format_general_summary(self, memories: List[Dict]) -> str:
        """Format a general summary of what we know."""
        if not memories:
            return "I don't have much information about you yet."
        
        summary_items = []
        for i, mem in enumerate(memories[:5]):  # Top 5 memories
            content = mem['content']
            summary_items.append(f"• {content}")
        
        return "Here's what I know about you:\n\n" + "\n".join(summary_items)
    
    def _format_relevant_memories(self, memories: List[Dict]) -> str:
        """Format relevant memories for display."""
        if not memories:
            return "I don't have relevant information about that."
        
        items = []
        for mem in memories:
            content = mem['content']
            items.append(f"You told me: {content}")
        
        return "Here's what I remember:\n\n" + "\n\n".join(items)
