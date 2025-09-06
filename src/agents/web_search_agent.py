"""
Web Search Agent - Simple, focused web searches without research overhead.
Handles basic web searches, image searches, and current information requests.
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any
from crewai import Agent

try:
    from .base_agent import AgentFactory
    from ..tools.web_browser_tool import WebBrowserTool, WebSearchTool
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from agents.base_agent import AgentFactory
    from tools.web_browser_tool import WebBrowserTool, WebSearchTool

logger = logging.getLogger(__name__)

class WebSearchAgentManager:
    """Manager for web search agent operations."""

    def __init__(self):
        self.web_browser = WebBrowserTool()
        self.web_search = WebSearchTool()

    def get_web_search_tools(self) -> List:
        """Get all tools available to the web search agent."""
        return [
            self.web_browser,
            self.web_search
        ]

    def create_agent(self, user_id: str = "default") -> Agent:
        """Create a web search agent with necessary tools."""
        tools = self.get_web_search_tools()
        return AgentFactory.create_web_search_agent(user_id, tools)

class WebSearchAgent:
    """Simple web search agent for basic searches and image requests."""
    
    def __init__(self):
        self.web_search_tool = None
        self._initialize_tools()
    
    def _initialize_tools(self):
        """Initialize web search tools."""
        try:
            from tools.web_browser_tool import WebSearchTool
            self.web_search_tool = WebSearchTool()
            logger.info("Web search agent initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize web search agent: {e}")
    
    async def execute_web_search(self, query: str, user_id: str, search_type: str = "general") -> str:
        """Execute a simple web search without research overhead."""
        try:
            logger.info(f"Executing {search_type} web search for user {user_id}: {query}")

            if search_type == "image":
                return await self._handle_image_search(query)
            elif search_type == "current":
                return await self._handle_current_info_search(query)
            else:
                return await self._handle_general_search(query)

        except Exception as e:
            logger.error(f"Web search execution failed: {e}")
            return f"I encountered an error while searching. Please try again: {str(e)}"
    
    async def _handle_image_search(self, query: str) -> str:
        """Handle image search requests - pure image focus."""
        try:
            from tools.query_parser import query_parser
            
            # Extract the subject from the query
            parse_result = query_parser.parse_image_query(query)
            subject = parse_result.get("subject", query)
            
            logger.info(f"Image search for subject: {subject}")
            
            # Use web search tool for images
            if self.web_search_tool:
                # Use async version to avoid context conflicts
                result = await self.web_search_tool.run_async(query)
                
                # Check if we got actual image results
                if any(indicator in result for indicator in ["Found image of", "Found images of", "Direct Image URL", "wikimedia.org", "unsplash.com", ".jpg", ".png"]):
                    return result
                
                # If no direct images, provide clean search links
                from urllib.parse import quote_plus
                encoded_query = quote_plus(subject)
                
                return f"""**Images of {subject}:**

🔍 [Google Images](https://www.google.com/search?q={encoded_query}&tbm=isch)
🔍 [Bing Images](https://www.bing.com/images/search?q={encoded_query})
🔍 [DuckDuckGo Images](https://duckduckgo.com/?q={encoded_query}&iax=images&ia=images)

Click any link to see images of {subject}."""
            
            else:
                return "Web search tool not available. Please try searching manually."
                
        except Exception as e:
            logger.error(f"Image search failed: {e}")
            return f"Image search encountered an error: {str(e)}"
    
    async def _handle_current_info_search(self, query: str) -> str:
        """Handle searches for current information (stock prices, news, etc.)."""
        try:
            logger.info(f"Current info search: {query}")

            # Use web search tool for current information
            if self.web_search_tool:
                # Use async version to avoid context conflicts
                result = await self.web_search_tool.run_async(query)
                return result
            else:
                return "Web search tool not available. Please try searching manually."

        except Exception as e:
            logger.error(f"Current info search failed: {e}")
            return f"Current info search encountered an error: {str(e)}"
    
    async def _handle_general_search(self, query: str) -> str:
        """Handle general web searches."""
        try:
            logger.info(f"General web search: {query}")

            # Use web search tool for general searches
            if self.web_search_tool:
                # Use async version to avoid context conflicts
                result = await self.web_search_tool.run_async(query)
                return result
            else:
                return "Web search tool not available. Please try searching manually."

        except Exception as e:
            logger.error(f"General web search failed: {e}")
            return f"Web search encountered an error: {str(e)}"

class WebSearchExecutor:
    """Executor for web search tasks."""
    
    def __init__(self):
        self.agent = WebSearchAgent()
    
    async def execute_search_task(self, query: str, user_id: str, search_type: str = "general") -> str:
        """Execute a web search task."""
        try:
            return await self.agent.execute_web_search(query, user_id, search_type)
        except Exception as e:
            logger.error(f"Web search task execution failed: {e}")
            return f"Search failed: {str(e)}"

# Global web search executor instance
web_search_executor = WebSearchExecutor()
