"""
Multi-Agent Conversation System
Agents discuss requests among themselves to determine the best approach.
"""

import logging
from typing import List, Dict, Any, Optional
import asyncio

logger = logging.getLogger(__name__)

class AgentConversation:
    """Facilitates conversations between agents to determine task assignment."""
    
    def __init__(self, llm_manager):
        self.llm_manager = llm_manager
        self.conversation_history = []
    
    async def facilitate_agent_discussion(self, user_request: str, available_agents: List[str]) -> Dict[str, Any]:
        """
        Let agents discuss the request and reach consensus on approach.
        """
        try:
            logger.info(f"Starting agent discussion for: {user_request}")
            
            # Phase 1: Each agent gives their initial assessment
            initial_assessments = await self._gather_initial_assessments(user_request, available_agents)
            
            # Phase 2: Agents discuss and refine their positions
            discussion_rounds = await self._conduct_discussion_rounds(user_request, initial_assessments)
            
            # Phase 3: Reach consensus on final approach
            consensus = await self._reach_consensus(user_request, discussion_rounds)
            
            return consensus
            
        except Exception as e:
            logger.error(f"Error in agent discussion: {e}")
            return self._fallback_consensus(user_request, available_agents)
    
    async def _gather_initial_assessments(self, user_request: str, available_agents: List[str]) -> List[Dict]:
        """Each agent provides their initial assessment of the request."""
        assessments = []
        
        for agent_name in available_agents:
            assessment = await self._get_agent_assessment(agent_name, user_request)
            assessments.append({
                'agent': agent_name,
                'assessment': assessment,
                'round': 1
            })
        
        return assessments
    
    async def _get_agent_assessment(self, agent_name: str, user_request: str) -> Dict:
        """Get a specific agent's assessment of the request."""
        
        agent_personas = {
            'research': {
                'role': 'Research Specialist',
                'expertise': 'Explaining concepts, analyzing topics, providing detailed explanations of how things work, educational content',
                'personality': 'Thorough, analytical, loves diving deep into topics and explaining them clearly'
            },
            'coding': {
                'role': 'Programming Expert',
                'expertise': 'Writing code, debugging, explaining programming concepts, technical solutions, software development',
                'personality': 'Logical, precise, focused on technical accuracy'
            },
            'web_search': {
                'role': 'Current Information Specialist',
                'expertise': 'Finding CURRENT and REAL-TIME information like stock prices, news, weather, images, live data from the web',
                'personality': 'Quick, resourceful, specializes in getting the latest information from the internet'
            },
            'general': {
                'role': 'Conversation Partner',
                'expertise': 'General chat, personal conversations, casual interactions, emotional support',
                'personality': 'Friendly, empathetic, great at casual conversation'
            }
        }
        
        agent_info = agent_personas.get(agent_name, agent_personas['general'])
        
        prompt = f"""You are the {agent_info['role']} agent.

REQUEST: "{user_request}"

As the {agent_info['role']}:
- Your expertise: {agent_info['expertise']}

Rate your ability to handle this request (1-10):
If this asks for CURRENT/REAL-TIME data (stock prices, news, weather), Web Search agent should handle it.
If this asks for EXPLANATIONS/CONCEPTS, Research agent should handle it.
If this involves CODE/PROGRAMMING, Coding agent should handle it.
If this is CASUAL CHAT, General agent should handle it.

Your rating (1-10) and brief reason:"""

        try:
            response = await self.llm_manager.generate_response(
                user_message=prompt,
                system_message=f"You are the {agent_info['role']} agent. Give a brief assessment.",
                max_tokens=100
            )
            
            return {
                'capability_score': self._extract_capability_score(response),
                'value_proposition': response,
                'collaboration_suggestions': self._extract_collaboration_suggestions(response),
                'recommended_approach': response
            }
            
        except Exception as e:
            logger.error(f"Error getting assessment from {agent_name}: {e}")
            return {
                'capability_score': 5,
                'value_proposition': f"I'm the {agent_info['role']} and I can help with this request.",
                'collaboration_suggestions': [],
                'recommended_approach': "I'll handle this to the best of my ability."
            }
    
    def _extract_capability_score(self, response: str) -> int:
        """Extract capability score from agent response."""
        import re
        
        # Look for patterns like "rate 1-10" or "score: 8" etc.
        patterns = [
            r'rate.*?(\d+)',
            r'score.*?(\d+)', 
            r'(\d+)/10',
            r'(\d+)\s*out of 10'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response.lower())
            if match:
                score = int(match.group(1))
                return min(10, max(1, score))  # Clamp between 1-10
        
        # Default score if no clear rating found
        return 5
    
    def _extract_collaboration_suggestions(self, response: str) -> List[str]:
        """Extract which other agents this agent suggests involving."""
        response_lower = response.lower()
        suggested_agents = []
        
        agent_keywords = {
            'research': ['research', 'analyze', 'explain', 'information'],
            'coding': ['code', 'program', 'technical', 'debug'],
            'web_search': ['search', 'find', 'current', 'web', 'image'],
            'general': ['chat', 'conversation', 'talk', 'general']
        }
        
        for agent, keywords in agent_keywords.items():
            if any(keyword in response_lower for keyword in keywords):
                suggested_agents.append(agent)
        
        return suggested_agents
    
    async def _conduct_discussion_rounds(self, user_request: str, initial_assessments: List[Dict]) -> List[Dict]:
        """Conduct discussion rounds where agents respond to each other."""
        discussion_history = initial_assessments.copy()
        
        # Conduct 2 rounds of discussion
        for round_num in range(2, 4):  # Rounds 2 and 3
            round_discussions = []
            
            for assessment in initial_assessments:
                agent_name = assessment['agent']
                
                # Show this agent what others have said
                other_opinions = [a for a in discussion_history if a['agent'] != agent_name]
                
                discussion_response = await self._get_agent_discussion_response(
                    agent_name, user_request, other_opinions, round_num
                )
                
                round_discussions.append({
                    'agent': agent_name,
                    'assessment': discussion_response,
                    'round': round_num
                })
            
            discussion_history.extend(round_discussions)
        
        return discussion_history
    
    async def _get_agent_discussion_response(self, agent_name: str, user_request: str, other_opinions: List[Dict], round_num: int) -> Dict:
        """Get an agent's response after hearing other agents' opinions."""
        
        other_opinions_text = ""
        for opinion in other_opinions[-4:]:  # Last 4 opinions to keep it manageable
            other_opinions_text += f"\n{opinion['agent']}: {opinion['assessment']['value_proposition'][:200]}..."
        
        prompt = f"""You are the {agent_name} agent. You've heard what other agents think about this request:

USER REQUEST: "{user_request}"

OTHER AGENTS' OPINIONS:{other_opinions_text}

Now that you've heard from your colleagues, what's your updated assessment?
- Do you still think you should handle this?
- Has anyone made a compelling case?
- Should you collaborate with someone?
- Any changes to your approach?

Keep your response concise and focused."""

        try:
            response = await self.llm_manager.generate_response(
                user_message=prompt,
                system_message=f"You are the {agent_name} agent in round {round_num} of team discussion.",
                max_tokens=200
            )
            
            return {
                'capability_score': self._extract_capability_score(response),
                'value_proposition': response,
                'collaboration_suggestions': self._extract_collaboration_suggestions(response),
                'recommended_approach': response
            }
            
        except Exception as e:
            logger.error(f"Error in discussion round for {agent_name}: {e}")
            return {
                'capability_score': 5,
                'value_proposition': "I maintain my original assessment.",
                'collaboration_suggestions': [],
                'recommended_approach': "I'll proceed as originally planned."
            }
    
    async def _reach_consensus(self, user_request: str, discussion_history: List[Dict]) -> Dict[str, Any]:
        """Analyze the discussion and reach consensus on the best approach."""
        
        # Get final scores for each agent
        final_scores = {}
        for discussion in discussion_history:
            if discussion['round'] == 3:  # Latest round
                agent = discussion['agent']
                score = discussion['assessment']['capability_score']
                final_scores[agent] = score
        
        # Find the agent(s) with highest confidence
        if final_scores:
            max_score = max(final_scores.values())
            top_agents = [agent for agent, score in final_scores.items() if score >= max_score - 1]
        else:
            top_agents = ['general']  # Fallback
        
        # Determine if collaboration is needed
        collaboration_needed = len(top_agents) > 1 or max_score < 8
        
        return {
            'primary_agent': top_agents[0],
            'collaborating_agents': top_agents[1:] if collaboration_needed else [],
            'confidence': max_score / 10.0,
            'approach': 'collaborative' if collaboration_needed else 'single_agent',
            'reasoning': f"Agents discussed and {top_agents[0]} scored highest ({max_score}/10)",
            'discussion_summary': self._summarize_discussion(discussion_history)
        }
    
    def _summarize_discussion(self, discussion_history: List[Dict]) -> str:
        """Create a brief summary of the agent discussion."""
        agent_positions = {}
        
        for discussion in discussion_history:
            if discussion['round'] == 3:  # Final positions
                agent = discussion['agent']
                score = discussion['assessment']['capability_score']
                agent_positions[agent] = score
        
        summary = "Agent discussion results: "
        for agent, score in sorted(agent_positions.items(), key=lambda x: x[1], reverse=True):
            summary += f"{agent}({score}/10) "
        
        return summary.strip()
    
    def _fallback_consensus(self, user_request: str, available_agents: List[str]) -> Dict[str, Any]:
        """Fallback consensus when discussion fails."""
        return {
            'primary_agent': 'general',
            'collaborating_agents': [],
            'confidence': 0.5,
            'approach': 'single_agent',
            'reasoning': 'Fallback due to discussion error',
            'discussion_summary': 'Discussion failed, using general agent'
        }
