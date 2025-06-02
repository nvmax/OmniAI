"""
Research agent for information gathering and web browsing.
Specializes in finding and synthesizing information from various sources.
"""

import logging
from typing import List, Optional
from crewai import Agent

try:
    from .base_agent import AgentFactory
    from ..tools.web_browser_tool import WebBrowserTool, WebSearchTool, LocalSearchTool
    from ..tools.vector_db_tool import VectorDBRetrievalTool, MemoryStorageTool
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from agents.base_agent import AgentFactory
    from tools.web_browser_tool import WebBrowserTool, WebSearchTool, LocalSearchTool
    from tools.vector_db_tool import VectorDBRetrievalTool, MemoryStorageTool

logger = logging.getLogger(__name__)

class ResearchAgentManager:
    """Manager for research agent operations."""
    
    def __init__(self):
        self.web_browser = WebBrowserTool()
        self.web_search = WebSearchTool()
        self.local_search = LocalSearchTool()
        self.memory_search = VectorDBRetrievalTool()
        self.memory_storage = MemoryStorageTool()
    
    def get_research_tools(self) -> List:
        """Get all tools available to the research agent."""
        return [
            self.web_browser,
            self.web_search,
            self.local_search,
            self.memory_search,
            self.memory_storage
        ]
    
    def create_research_agent(self, user_id: str = "default") -> Agent:
        """Create a research agent with all necessary tools."""
        tools = self.get_research_tools()
        return AgentFactory.create_research_agent(user_id, tools)
    
    def analyze_research_query(self, query: str) -> dict:
        """Analyze a research query to determine the best approach."""
        query_lower = query.lower()
        
        analysis = {
            "needs_web_search": False,
            "needs_local_search": False,
            "needs_memory_search": True,  # Always check memory first
            "query_type": "general",
            "suggested_tools": []
        }
        
        # Determine if web search is needed
        web_indicators = [
            "latest", "recent", "current", "news", "today", "2024", "2025", "2023",
            "what is", "who is", "when did", "where is", "how to",
            "search for", "find information", "look up", "stock price", "price of",
            "real-time", "live", "now", "as of", "today's",
            "picture", "image", "photo", "pic", "show me", "find a picture"
        ]
        
        if any(indicator in query_lower for indicator in web_indicators):
            analysis["needs_web_search"] = True
            analysis["suggested_tools"].append("web_browser")
        
        # Determine if local search might be helpful
        local_indicators = [
            "programming", "python", "javascript", "discord", "bot",
            "ai", "machine learning", "code", "development"
        ]
        
        if any(indicator in query_lower for indicator in local_indicators):
            analysis["needs_local_search"] = True
            analysis["suggested_tools"].append("local_search")
        
        # Always suggest memory search
        analysis["suggested_tools"].append("memory_search")
        
        # Determine query type
        if any(word in query_lower for word in ["code", "programming", "function", "script"]):
            analysis["query_type"] = "technical"
        elif any(word in query_lower for word in ["how to", "tutorial", "guide", "steps"]):
            analysis["query_type"] = "instructional"
        elif any(word in query_lower for word in ["what is", "define", "explain", "meaning"]):
            analysis["query_type"] = "definitional"
        
        return analysis

class ResearchTaskExecutor:
    """Executes research tasks using the research agent."""
    
    def __init__(self):
        self.agent_manager = ResearchAgentManager()
    
    async def execute_research_task(self, query: str, user_id: str = "default") -> str:
        """Execute a research task and return comprehensive results."""
        try:
            # Analyze the query
            analysis = self.agent_manager.analyze_research_query(query)
            
            # Create research agent
            research_agent = self.agent_manager.create_research_agent(user_id)
            
            # Build research strategy based on analysis
            research_strategy = self._build_research_strategy(query, analysis)
            
            # Execute research using the agent
            # Note: In a full implementation, you would use CrewAI's task execution
            # For now, we'll simulate the research process
            
            results = []
            
            # Check memory first
            if analysis["needs_memory_search"]:
                memory_tool = self.agent_manager.memory_search
                memory_results = memory_tool._run(query, user_id)
                if "No relevant information found" not in memory_results:
                    results.append(f"From Memory:\n{memory_results}")
            
            # Check local knowledge
            if analysis["needs_local_search"]:
                local_tool = self.agent_manager.local_search
                local_results = local_tool._run(query)
                if "No local information found" not in local_results:
                    results.append(f"Local Knowledge:\n{local_results}")
            
            # Perform web search if needed
            if analysis["needs_web_search"]:
                try:
                    web_search_tool = self.agent_manager.web_search
                    web_results = web_search_tool._run(query)
                    if web_results and "Error" not in web_results and "failed" not in web_results.lower():
                        results.append(f"Web Search Results:\n{web_results}")
                    else:
                        results.append(f"Web search encountered an issue: {web_results}")
                except Exception as e:
                    logger.error(f"Web search error: {e}")
                    results.append(f"Web search failed: {str(e)}")
            
            # Compile final response
            if results:
                final_response = "\n\n".join(results)
                
                # Store important findings in memory
                if len(final_response) > 100:  # Only store substantial findings
                    storage_tool = self.agent_manager.memory_storage
                    storage_tool._run(
                        f"Research on '{query}': {final_response[:500]}...",
                        user_id,
                        "research"
                    )
                
                return final_response
            else:
                return f"I couldn't find specific information about '{query}'. Would you like me to search the web for more details?"
                
        except Exception as e:
            logger.error(f"Error executing research task: {e}")
            return f"I encountered an error while researching '{query}': {str(e)}"
    
    def _build_research_strategy(self, query: str, analysis: dict) -> str:
        """Build a research strategy based on query analysis."""
        strategy_parts = []
        
        strategy_parts.append(f"Research Strategy for: '{query}'")
        strategy_parts.append(f"Query Type: {analysis['query_type']}")
        
        if analysis["needs_memory_search"]:
            strategy_parts.append("1. Search conversation history and stored knowledge")
        
        if analysis["needs_local_search"]:
            strategy_parts.append("2. Check local knowledge base for relevant information")
        
        if analysis["needs_web_search"]:
            strategy_parts.append("3. Perform web search for current/additional information")
        
        strategy_parts.append("4. Synthesize findings and provide comprehensive response")
        
        return "\n".join(strategy_parts)

# Global research manager instance
research_manager = ResearchAgentManager()
research_executor = ResearchTaskExecutor()
