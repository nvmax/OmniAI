"""
Base agent configuration for Omni-Assistant.
Provides common functionality and LLM integration for all agents.
"""

import logging
from typing import Optional, Dict, Any, List
from crewai import Agent
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun

try:
    from ..llm_integration import llm_manager
    from ..personalities import personality_manager
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from llm_integration import llm_manager
    from personalities import personality_manager

logger = logging.getLogger(__name__)

class LocalLLM(LLM):
    """Custom LangChain LLM wrapper for LM Studio integration."""

    def __init__(self):
        super().__init__()
        self._llm_manager = llm_manager
    
    @property
    def _llm_type(self) -> str:
        return "local_lm_studio"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call the LM Studio LLM."""
        try:
            import asyncio
            
            # Run async function in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            response = loop.run_until_complete(
                self._llm_manager.generate_response(
                    user_message=prompt,
                    **kwargs
                )
            )
            
            loop.close()
            
            return response or "I apologize, but I'm having trouble generating a response right now."
            
        except Exception as e:
            logger.error(f"Error in LocalLLM call: {e}")
            return f"Error generating response: {str(e)}"

class BaseAgentConfig:
    """Base configuration for all agents."""

    def __init__(self, user_id: str = "default"):
        self.user_id = user_id
        self.llm = self._create_llm()
        self.personality = personality_manager.get_user_personality(user_id)

    def _create_llm(self):
        """Create an LLM instance that works with CrewAI."""
        try:
            # For now, just use LocalLLM to avoid dependency issues
            # TODO: Add proper LangChain LLM integration when needed
            return LocalLLM()

        except Exception as e:
            logger.warning(f"Failed to create LLM: {e}")
            return LocalLLM()
    
    def create_agent(
        self,
        role: str,
        goal: str,
        backstory: str,
        tools: List = None,
        verbose: bool = True,
        allow_delegation: bool = False,
        **kwargs
    ) -> Agent:
        """Create a CrewAI agent with common configuration."""
        
        # Incorporate personality into the backstory
        personality_context = f"\n\nPersonality Context: {self.personality.system_message}"
        enhanced_backstory = backstory + personality_context
        
        agent = Agent(
            role=role,
            goal=goal,
            backstory=enhanced_backstory,
            llm=self.llm,
            tools=tools or [],
            verbose=verbose,
            allow_delegation=allow_delegation,
            **kwargs
        )
        
        return agent
    
    def get_system_message(self) -> str:
        """Get the system message for the current personality."""
        return self.personality.system_message

class AgentFactory:
    """Factory for creating configured agents."""
    
    @staticmethod
    def create_research_agent(user_id: str = "default", tools: List = None) -> Agent:
        """Create a research agent."""
        config = BaseAgentConfig(user_id)
        
        return config.create_agent(
            role="Research Specialist",
            goal=(
                "Conduct thorough research on topics, gather information from various sources, "
                "and provide comprehensive, accurate answers to user questions. Focus on "
                "finding reliable information and presenting it clearly."
            ),
            backstory=(
                "You are an expert research specialist with access to web browsing capabilities "
                "and a comprehensive knowledge base. You excel at finding relevant information, "
                "fact-checking, and synthesizing data from multiple sources. You always strive "
                "for accuracy and provide well-structured, informative responses."
            ),
            tools=tools or [],
            verbose=True,
            allow_delegation=False
        )
    
    @staticmethod
    def create_coding_agent(user_id: str = "default", tools: List = None) -> Agent:
        """Create a coding agent."""
        config = BaseAgentConfig(user_id)
        
        return config.create_agent(
            role="Senior Software Developer",
            goal=(
                "Help users with programming tasks including code generation, debugging, "
                "code review, and technical explanations. Provide clean, efficient, and "
                "well-documented code solutions."
            ),
            backstory=(
                "You are a senior software developer with extensive experience across multiple "
                "programming languages and frameworks. You excel at writing clean, efficient code, "
                "debugging complex issues, and explaining technical concepts clearly. You follow "
                "best practices and always consider code maintainability and performance."
            ),
            tools=tools or [],
            verbose=True,
            allow_delegation=False
        )
    
    @staticmethod
    def create_web_search_agent(user_id: str = "default", tools: List = None) -> Agent:
        """Create a web search agent."""
        config = BaseAgentConfig(user_id)

        return config.create_agent(
            role="Web Search Specialist",
            goal=(
                "Find current, accurate information from the web quickly and efficiently. "
                "Focus on real-time data, current events, prices, weather, and other "
                "time-sensitive information that requires web searches."
            ),
            backstory=(
                "You are a web search specialist with access to real-time web browsing and "
                "search capabilities. You excel at finding current information, verifying "
                "facts from multiple sources, and providing specific details like addresses, "
                "phone numbers, ratings, and prices. You focus on delivering accurate, "
                "up-to-date information quickly."
            ),
            tools=tools or [],
            verbose=True,
            allow_delegation=False
        )

    @staticmethod
    def create_image_analysis_agent(user_id: str = "default", tools: List = None) -> Agent:
        """Create an image analysis agent."""
        config = BaseAgentConfig(user_id)

        return config.create_agent(
            role="Image Analysis Specialist",
            goal=(
                "Analyze and understand visual content in images. Provide detailed descriptions, "
                "identify objects and scenes, offer artistic commentary, and extract meaningful "
                "information from visual data."
            ),
            backstory=(
                "You are an image analysis specialist with advanced computer vision capabilities. "
                "You can analyze images to identify objects, people, scenes, text, and artistic "
                "elements. You provide detailed descriptions, technical analysis, and insightful "
                "commentary about visual content, adapting your analysis style to match the "
                "user's personality and needs."
            ),
            tools=tools or [],
            verbose=True,
            allow_delegation=False
        )

    @staticmethod
    def create_general_agent(user_id: str = "default", tools: List = None) -> Agent:
        """Create a general-purpose agent."""
        config = BaseAgentConfig(user_id)

        return config.create_agent(
            role="General Assistant",
            goal=(
                "Provide helpful, accurate, and engaging responses to a wide variety of user "
                "questions and requests. Adapt to different topics and maintain a helpful, "
                "friendly demeanor while being informative and useful."
            ),
            backstory=(
                "You are a knowledgeable and versatile AI assistant capable of helping with "
                "a wide range of topics including general questions, explanations, creative "
                "tasks, and problem-solving. You adapt your communication style to match "
                "the user's needs and always strive to be helpful and informative."
            ),
            tools=tools or [],
            verbose=True,
            allow_delegation=True
        )

def get_agent_for_task(task_type: str, user_id: str = "default", tools: List = None) -> Agent:
    """Get the appropriate agent for a specific task type."""
    
    task_type_lower = task_type.lower()
    
    if any(keyword in task_type_lower for keyword in ["research", "search", "find", "lookup", "information"]):
        return AgentFactory.create_research_agent(user_id, tools)
    elif any(keyword in task_type_lower for keyword in ["code", "programming", "debug", "script", "function"]):
        return AgentFactory.create_coding_agent(user_id, tools)
    else:
        return AgentFactory.create_general_agent(user_id, tools)
