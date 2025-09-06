"""
CrewAI orchestrator for managing multi-agent tasks and workflows.
Coordinates between different agents based on user requests and context.
"""

import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any
from crewai import Crew, Task, Process
from crewai.agent import Agent

try:
    from .memory_manager import memory_manager
    from .personalities import personality_manager
    from .llm_integration import llm_manager
    from .semantic_memory import SemanticMemoryHandler
    from .agent_conversation import AgentConversation
    from .agents.base_agent import AgentFactory
    from .agents.research_agent import ResearchAgentManager
    from .agents.coding_agent import CodingAgentManager
    from .agents.web_search_agent import WebSearchAgentManager
    from .agents.image_analysis_agent import ImageAnalysisAgentManager
except ImportError:
    from memory_manager import memory_manager
    from personalities import personality_manager
    from llm_integration import llm_manager
    from semantic_memory import SemanticMemoryHandler
    from agent_conversation import AgentConversation
    from agents.base_agent import AgentFactory
    from agents.research_agent import ResearchAgentManager
    from agents.coding_agent import CodingAgentManager
    from agents.web_search_agent import WebSearchAgentManager
    from agents.image_analysis_agent import ImageAnalysisAgentManager

logger = logging.getLogger(__name__)

class IntelligentRouter:
    """
    Intelligent router that uses LLM-based natural language understanding
    to determine the most appropriate agent for handling user requests.
    """

    def __init__(self):
        self.routing_cache = {}

        # Define available agents and their capabilities
        self.agents = {
            "research": {
                "description": "Specializes in research, analysis, explanations, and finding detailed information",
                "capabilities": [
                    "Conducting thorough research on topics",
                    "Providing detailed explanations and analysis",
                    "Comparing and contrasting concepts",
                    "Investigating complex questions",
                    "Academic and scientific research"
                ],
                "examples": [
                    "Explain quantum computing",
                    "Research the history of artificial intelligence",
                    "What are the pros and cons of renewable energy?",
                    "Analyze the economic impact of remote work"
                ]
            },
            "coding": {
                "description": "Specializes in programming, software development, debugging, and technical solutions",
                "capabilities": [
                    "Writing code in various programming languages",
                    "Debugging and fixing code issues",
                    "Explaining programming concepts",
                    "Code review and optimization",
                    "API integration and database queries"
                ],
                "examples": [
                    "Write a Python function to sort a list",
                    "Debug this JavaScript error",
                    "Create a REST API endpoint",
                    "Optimize this SQL query"
                ]
            },
            "web_search": {
                "description": "Specializes in finding current information, quick lookups, real-time data, and finding images on the web",
                "capabilities": [
                    "Finding current news and events",
                    "Looking up stock prices and market data",
                    "Quick fact checking",
                    "Finding recent information",
                    "Simple web searches",
                    "Finding images and pictures on the web",
                    "Image search and discovery",
                    "Visual content lookup"
                ],
                "examples": [
                    "What's the current price of Bitcoin?",
                    "Find recent news about climate change",
                    "Look up the weather in Tokyo",
                    "Search for the latest iPhone release",
                    "Find me a picture of the rock",
                    "Show me images of cats",
                    "Find a photo of the Eiffel Tower"
                ]
            },
            "general": {
                "description": "Handles general conversation, casual chat, and requests that don't require specialized tools",
                "capabilities": [
                    "General conversation and chat",
                    "Answering simple questions",
                    "Providing basic information",
                    "Casual interactions",
                    "Default fallback responses"
                ],
                "examples": [
                    "Hello, how are you?",
                    "Tell me a joke",
                    "What's your favorite color?",
                    "Thanks for your help"
                ]
            }
        }

    async def route_request(self, user_message: str, user_id: str = None) -> Dict[str, Any]:
        """
        Use LLM-based natural language understanding to route the request
        to the most appropriate agent.
        """
        # Check cache first
        cache_key = f"{user_id}:{hash(user_message.lower().strip())}"
        if cache_key in self.routing_cache:
            return self.routing_cache[cache_key]

        # Create routing prompt for the LLM
        routing_prompt = self._create_routing_prompt(user_message)

        try:
            # Import LLM manager here to avoid circular imports
            try:
                from .llm_integration import llm_manager
            except ImportError:
                from llm_integration import llm_manager

            # Get routing decision from LLM
            routing_response = await llm_manager.generate_response(
                user_message=routing_prompt,
                system_message="You are an intelligent request router. Analyze the user's request and determine the best agent to handle it. Respond only with valid JSON.",
                max_tokens=300
            )

            if routing_response:
                # Parse the LLM response
                routing_decision = self._parse_routing_response(routing_response, user_message)

                # Cache the result
                self.routing_cache[cache_key] = routing_decision
                return routing_decision

        except Exception as e:
            logger.error(f"Error in LLM-based routing: {e}")

        # Fallback to simple heuristics if LLM routing fails
        return self._fallback_routing(user_message)

    def _create_routing_prompt(self, user_message: str) -> str:
        """Create a prompt for the LLM to make routing decisions."""

        agents_info = ""
        for agent_name, info in self.agents.items():
            agents_info += f"\n{agent_name}:\n"
            agents_info += f"  Description: {info['description']}\n"
            agents_info += f"  Examples: {', '.join(info['examples'][:2])}\n"

        prompt = f"""Analyze this user request and determine which agent should handle it:

USER REQUEST: "{user_message}"

AVAILABLE AGENTS:{agents_info}

Respond with ONLY a JSON object in this exact format:
{{
    "agent": "agent_name",
    "confidence": 0.85,
    "reasoning": "Brief explanation of why this agent was chosen"
}}

Choose the agent that best matches the user's intent and needs."""

        return prompt

    def _parse_routing_response(self, response: str, user_message: str) -> Dict[str, Any]:
        """Parse the LLM routing response into a structured decision."""
        try:
            import json
            import re

            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                parsed = json.loads(json_str)

                agent = parsed.get("agent", "general")
                confidence = float(parsed.get("confidence", 0.5))
                reasoning = parsed.get("reasoning", "Default routing")

                # Validate agent exists
                if agent not in self.agents:
                    agent = "general"

                return {
                    "primary_task": agent,
                    "confidence": confidence,
                    "reasoning": reasoning,
                    "method": "llm_routing"
                }

        except Exception as e:
            logger.error(f"Error parsing routing response: {e}")

        # Return default if parsing fails, but check for image requests using user message
        return self._fallback_routing_with_context(user_message)

    def _fallback_routing_with_context(self, user_message: str) -> Dict[str, Any]:
        """Provide intelligent fallback routing when JSON parsing fails."""
        # Check the user message for image requests
        message_lower = user_message.lower()
        if any(word in message_lower for word in ["image", "picture", "photo", "pic", "show me", "find me"]):
            logger.info(f"Fallback routing detected image request in: '{user_message}'")
            return {
                "primary_task": "web_search",
                "confidence": 0.8,
                "reasoning": "Fallback routing detected image request",
                "method": "fallback_heuristic"
            }

        # Check for other patterns
        if any(word in message_lower for word in ["search", "find", "look up", "google"]):
            return {
                "primary_task": "web_search",
                "confidence": 0.7,
                "reasoning": "Fallback routing detected search request",
                "method": "fallback_heuristic"
            }

        # Default fallback
        return {
            "primary_task": "general",
            "confidence": 0.5,
            "reasoning": "Fallback due to parsing error",
            "method": "fallback"
        }

    def _fallback_routing(self, user_message: str) -> Dict[str, Any]:
        """Simple fallback routing using basic heuristics."""
        message_lower = user_message.lower().strip()

        # Simple keyword-based fallback
        if any(word in message_lower for word in ["code", "program", "debug", "python", "javascript"]):
            return {"primary_task": "coding", "confidence": 0.7, "reasoning": "Contains coding keywords", "method": "fallback"}
        elif any(word in message_lower for word in ["search", "find", "current", "latest", "price"]):
            return {"primary_task": "web_search", "confidence": 0.7, "reasoning": "Contains search keywords", "method": "fallback"}
        elif any(word in message_lower for word in ["explain", "research", "analyze", "what is", "how does"]):
            return {"primary_task": "research", "confidence": 0.7, "reasoning": "Contains research keywords", "method": "fallback"}
        elif any(word in message_lower for word in ["image", "picture", "photo"]) and any(word in message_lower for word in ["find", "show", "search"]):
            return {"primary_task": "image_search", "confidence": 0.7, "reasoning": "Contains image search keywords", "method": "fallback"}
        else:
            return {"primary_task": "general", "confidence": 0.6, "reasoning": "Default general conversation", "method": "fallback"}

    # Legacy method for backward compatibility
    def classify_request(self, user_message: str, user_id: str = None) -> Dict[str, Any]:
        """Legacy method that calls the new async routing method."""
        import asyncio

        try:
            # Try to get the current event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an async context, we can't use asyncio.run()
                # This is a limitation - we'll fall back to simple routing
                return self._fallback_routing(user_message)
            else:
                # If no loop is running, we can create one
                return asyncio.run(self.route_request(user_message, user_id))
        except RuntimeError:
            # Fallback if async doesn't work
            return self._fallback_routing(user_message)

class MultiAgentCrewOrchestrator:
    """
    Proper CrewAI-based orchestrator that uses actual multi-agent workflows.
    Coordinates specialized agents to work together on complex tasks.
    """

    def __init__(self):
        self.router = IntelligentRouter()  # Keep as fallback
        self.semantic_memory = SemanticMemoryHandler(memory_manager)
        self.agent_conversation = AgentConversation(llm_manager)
        self.agent_managers = {
            'research': ResearchAgentManager(),
            'coding': CodingAgentManager(),
            'web_search': WebSearchAgentManager(),
            'image_analysis': ImageAnalysisAgentManager()
        }
        self.active_crews = {}  # Cache for active crew instances

    async def process_request(
        self,
        user_message: str,
        user_id: str,
        channel_id: str = None
    ) -> str:
        """Process a user request using proper multi-agent CrewAI workflows."""
        try:
            # Get user's personality context
            personality = personality_manager.get_user_personality(user_id)
            logger.info(f"Using personality '{personality.name}' for user {user_id}")

            # Handle memory requests using semantic similarity (no hardcoded patterns)
            if self._is_memory_request(user_message):
                logger.info(f"Detected memory request: {user_message}")
                return await self.semantic_memory.handle_memory_request(user_message, user_id)

            # Enhance coding requests with explicit formatting instructions
            if any(word in user_message.lower() for word in ['write', 'code', 'function', 'program', 'script']):
                user_message = f"{user_message}\n\nIMPORTANT: Please ensure all code is properly indented (4 spaces for Python) and formatted correctly in markdown code blocks."

            # Check if this is a simple greeting or casual message
            if self._is_simple_greeting_or_casual(user_message):
                logger.info(f"Detected simple greeting/casual message: {user_message}")
                # Handle as general conversation without routing to specialized agents
                context = memory_manager.get_intelligent_context(user_id, user_message)
                response = await self._fallback_to_llm(user_message, user_id, context)
                await self._update_memory(user_message, response, user_id, channel_id)
                return self._clean_final_response(response)

            # Check if this is a simple personal statement that should be handled conversationally
            if self._is_simple_personal_statement(user_message):
                logger.info(f"Detected simple personal statement: {user_message}")
                # Handle as general conversation without routing to specialized agents
                context = memory_manager.get_intelligent_context(user_id, user_message)
                response = await self._fallback_to_llm(user_message, user_id, context)
                await self._update_memory(user_message, response, user_id, channel_id)
                return self._clean_final_response(response)

            # Get intelligent conversation context
            context = memory_manager.get_intelligent_context(user_id, user_message)

            # Use simple rule-based routing for clear cases, agent discussion for complex ones
            routing_decision = await self._smart_route_request(user_message, user_id)

            # Execute the task using the selected agent
            response = await self._execute_single_agent_task(
                user_message, user_id, routing_decision, context
            )

            if not response or "technical difficulties" in response:
                return "I'm having trouble processing your request. Please try again in a moment."

            # Clean the response
            response = self._clean_final_response(response)
            logger.info(f"Response generated successfully: {len(response)} characters")

            # Fix code formatting in response
            logger.info(f"Before code formatting fix: {response[:200]}...")
            response = self._fix_code_formatting(response)
            logger.info(f"After code formatting fix: {response[:200]}...")

            # Update memory with conversation
            await self._update_memory(user_message, response, user_id, channel_id)

            return response

        except Exception as e:
            logger.error(f"Error processing request: {e}")
            return "I apologize, but I encountered an error while processing your request. Please try again."

    def _analyze_task_complexity(self, user_message: str, routing_decision: Dict) -> Dict:
        """Analyze if a task requires multi-agent collaboration."""
        message_lower = user_message.lower()
        primary_task = routing_decision.get('primary_task', 'general')

        # Simple image search requests should NOT require collaboration
        is_simple_image_search = (
            primary_task == 'web_search' and
            any(word in message_lower for word in ['picture', 'image', 'photo', 'pic']) and
            any(word in message_lower for word in ['find', 'show', 'get']) and
            not any(word in message_lower for word in ['analyze', 'compare', 'explain', 'research'])
        )

        # Simple web search requests should NOT require collaboration
        is_simple_web_search = (
            primary_task == 'web_search' and
            not any(word in message_lower for word in ['analyze', 'compare', 'explain', 'research', 'code'])
        )

        # If it's a simple search, don't require collaboration
        if is_simple_image_search or is_simple_web_search:
            return {
                'requires_collaboration': False,
                'indicators': {
                    'research_and_code': False,
                    'web_search_and_analysis': False,
                    'multi_step_task': False,
                    'comprehensive_request': False,
                    'multiple_domains': False
                },
                'confidence': routing_decision.get('confidence', 0.5),
                'primary_agent': primary_task
            }

        # Indicators of complex tasks requiring collaboration
        complexity_indicators = {
            'research_and_code': any(word in message_lower for word in ['research', 'analyze', 'explain']) and
                               any(word in message_lower for word in ['code', 'implement', 'build', 'create']),
            'web_search_and_analysis': any(word in message_lower for word in ['find', 'search', 'current']) and
                                     any(word in message_lower for word in ['analyze', 'compare', 'explain']),
            'multi_step_task': any(phrase in message_lower for phrase in ['step by step', 'first', 'then', 'after that', 'finally']),
            'comprehensive_request': len(user_message.split()) > 20 or '?' in user_message and len(user_message.split('?')) > 2,
            'multiple_domains': sum(1 for domain in ['code', 'research', 'search', 'image']
                                  if any(word in message_lower for word in self._get_domain_keywords(domain))) > 1
        }

        requires_collaboration = any(complexity_indicators.values())

        return {
            'requires_collaboration': requires_collaboration,
            'indicators': complexity_indicators,
            'confidence': routing_decision.get('confidence', 0.5),
            'primary_agent': primary_task
        }

    def _get_domain_keywords(self, domain: str) -> List[str]:
        """Get keywords for different domains."""
        domain_keywords = {
            'code': ['code', 'program', 'script', 'function', 'debug', 'implement', 'build'],
            'research': ['research', 'analyze', 'explain', 'study', 'investigate', 'compare'],
            'search': ['find', 'search', 'current', 'latest', 'news', 'price', 'weather'],
            'image': ['image', 'picture', 'photo', 'visual', 'analyze image', 'show me']
        }
        return domain_keywords.get(domain, [])

    async def _execute_crew_workflow(
        self,
        user_message: str,
        user_id: str,
        routing_decision: Dict,
        context: Dict,
        task_complexity: Dict
    ) -> str:
        """Execute a multi-agent crew workflow for complex tasks."""
        try:
            # Create task breakdown
            tasks = await self._create_task_breakdown(user_message, routing_decision, task_complexity)

            # Create crew with appropriate agents
            crew = await self._create_crew_for_tasks(tasks, user_id)

            # Execute the crew workflow
            result = await self._execute_crew(crew, tasks, context)

            return result

        except Exception as e:
            logger.error(f"Error in crew workflow: {e}")
            # Fallback to single agent
            return await self._execute_single_agent_task(user_message, user_id, routing_decision, context)

    async def _execute_collaborative_workflow(
        self,
        user_message: str,
        user_id: str,
        agent_consensus: Dict,
        context: Dict
    ) -> str:
        """Execute a collaborative workflow based on agent consensus."""
        try:
            primary_agent = agent_consensus['primary_agent']
            collaborating_agents = agent_consensus['collaborating_agents']

            logger.info(f"Collaborative workflow: {primary_agent} leading, collaborating with {collaborating_agents}")

            # Start with primary agent
            primary_response = await self._get_agent_response(primary_agent, user_message, user_id, context)

            # If collaboration is needed, get input from other agents
            if collaborating_agents:
                collaborative_responses = []
                for agent in collaborating_agents:
                    collab_response = await self._get_agent_response(agent, user_message, user_id, context)
                    collaborative_responses.append(f"{agent}: {collab_response}")

                # Synthesize responses
                final_response = await self._synthesize_collaborative_responses(
                    user_message, primary_response, collaborative_responses
                )
                return final_response
            else:
                return primary_response

        except Exception as e:
            logger.error(f"Error in collaborative workflow: {e}")
            # Fallback to primary agent only
            routing_decision = {'primary_task': agent_consensus['primary_agent']}
            return await self._execute_single_agent_task(user_message, user_id, routing_decision, context)

    async def _get_agent_response(self, agent_type: str, user_message: str, user_id: str, context: Dict) -> str:
        """Get response from a specific agent."""
        try:
            if agent_type in self.agent_managers:
                agent_manager = self.agent_managers[agent_type]
                agent = agent_manager.create_agent(user_id)

                # Execute the agent task
                result = await agent.execute_task(user_message, context)
                return result if result else f"Agent {agent_type} completed the task."
            else:
                # Fallback to LLM for general agent
                return await self._fallback_to_llm(user_message, user_id, context)

        except Exception as e:
            logger.error(f"Error getting response from {agent_type}: {e}")
            return f"Agent {agent_type} encountered an error."

    async def _synthesize_collaborative_responses(
        self,
        user_message: str,
        primary_response: str,
        collaborative_responses: List[str]
    ) -> str:
        """Synthesize multiple agent responses into a coherent final response."""
        try:
            synthesis_prompt = f"""You need to synthesize multiple agent responses into one coherent answer.

USER REQUEST: "{user_message}"

PRIMARY AGENT RESPONSE:
{primary_response}

COLLABORATIVE AGENT RESPONSES:
{chr(10).join(collaborative_responses)}

Create a single, coherent response that combines the best insights from all agents. Keep it natural and conversational."""

            synthesized = await llm_manager.generate_response(
                user_message=synthesis_prompt,
                system_message="You are synthesizing multiple agent responses into one coherent answer.",
                max_tokens=500
            )

            return synthesized if synthesized else primary_response

        except Exception as e:
            logger.error(f"Error synthesizing responses: {e}")
            return primary_response  # Fallback to primary response

    async def _execute_single_agent_task(
        self,
        user_message: str,
        user_id: str,
        routing_decision: Dict,
        context: Dict
    ) -> str:
        """Execute a task using a single specialized agent."""
        try:
            agent_type = routing_decision.get('primary_task', 'general')

            # Get the appropriate agent
            if agent_type in self.agent_managers:
                agent_manager = self.agent_managers[agent_type]
                agent = agent_manager.create_agent(user_id)

                # Create enhanced system message with context
                enhanced_message = self._create_enhanced_system_message_for_agent(
                    agent_type, context, user_message, user_id
                )

                # Execute the task
                response = await self._execute_agent_task(agent, user_message, enhanced_message)
                return response
            else:
                # Fallback to general LLM response
                return await self._fallback_to_llm(user_message, user_id, context)

        except Exception as e:
            logger.error(f"Error in single agent task: {e}")
            return await self._fallback_to_llm(user_message, user_id, context)

    async def _create_task_breakdown(self, user_message: str, routing_decision: Dict, task_complexity: Dict) -> List[Dict]:
        """Break down complex requests into subtasks."""
        tasks = []

        # Analyze the request to identify subtasks
        if task_complexity['indicators'].get('research_and_code'):
            tasks.extend([
                {'type': 'research', 'description': f"Research and analyze: {user_message}", 'priority': 1},
                {'type': 'coding', 'description': f"Implement solution based on research: {user_message}", 'priority': 2}
            ])
        elif task_complexity['indicators'].get('web_search_and_analysis'):
            tasks.extend([
                {'type': 'web_search', 'description': f"Find current information: {user_message}", 'priority': 1},
                {'type': 'research', 'description': f"Analyze and synthesize findings: {user_message}", 'priority': 2}
            ])
        elif task_complexity['indicators'].get('multi_step_task'):
            # Parse multi-step instructions
            steps = self._parse_multi_step_request(user_message)
            for i, step in enumerate(steps):
                task_type = self._determine_task_type_for_step(step)
                tasks.append({
                    'type': task_type,
                    'description': step,
                    'priority': i + 1
                })
        else:
            # Default single task
            primary_type = routing_decision.get('primary_task', 'general')
            tasks.append({
                'type': primary_type,
                'description': user_message,
                'priority': 1
            })

        return tasks

    def _parse_multi_step_request(self, user_message: str) -> List[str]:
        """Parse a multi-step request into individual steps."""
        # Simple parsing - can be enhanced with NLP
        step_indicators = ['first', 'then', 'next', 'after that', 'finally', 'lastly']
        sentences = user_message.split('.')

        steps = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and (any(indicator in sentence.lower() for indicator in step_indicators) or len(steps) == 0):
                steps.append(sentence)

        return steps if steps else [user_message]

    def _determine_task_type_for_step(self, step: str) -> str:
        """Determine the appropriate agent type for a step."""
        step_lower = step.lower()

        if any(word in step_lower for word in ['code', 'program', 'implement', 'build']):
            return 'coding'
        elif any(word in step_lower for word in ['search', 'find', 'current', 'latest']):
            return 'web_search'
        elif any(word in step_lower for word in ['research', 'analyze', 'explain', 'study']):
            return 'research'
        elif any(word in step_lower for word in ['image', 'picture', 'photo']):
            return 'image_analysis'
        else:
            return 'general'

    async def _create_crew_for_tasks(self, tasks: List[Dict], user_id: str) -> 'Crew':
        """Create a CrewAI crew with agents needed for the tasks."""
        try:
            # Import CrewAI components
            from crewai import Crew, Task as CrewTask, Process

            # Determine unique agent types needed
            agent_types = list(set(task['type'] for task in tasks))

            # Create agents
            agents = []
            crew_tasks = []

            for agent_type in agent_types:
                if agent_type in self.agent_managers:
                    agent = self.agent_managers[agent_type].create_agent(user_id)
                    agents.append(agent)
                else:
                    # Create general agent as fallback
                    agent = AgentFactory.create_general_agent(user_id)
                    agents.append(agent)

            # Create CrewAI tasks
            for task_info in tasks:
                task_agent = next((agent for agent in agents
                                 if self._agent_matches_type(agent, task_info['type'])), agents[0])

                crew_task = CrewTask(
                    description=task_info['description'],
                    agent=task_agent,
                    expected_output="A comprehensive response addressing the user's request"
                )
                crew_tasks.append(crew_task)

            # Create the crew
            crew = Crew(
                agents=agents,
                tasks=crew_tasks,
                process=Process.sequential,  # Execute tasks in order
                verbose=True
            )

            return crew

        except ImportError:
            logger.error("CrewAI not properly installed")
            raise
        except Exception as e:
            logger.error(f"Error creating crew: {e}")
            raise

    def _agent_matches_type(self, agent: Agent, task_type: str) -> bool:
        """Check if an agent matches a task type."""
        # Simple matching based on agent role
        agent_role = agent.role.lower()

        type_mappings = {
            'research': ['research', 'analyst'],
            'coding': ['developer', 'programmer', 'software'],
            'web_search': ['search', 'web'],
            'image_analysis': ['image', 'visual'],
            'general': ['assistant', 'general']
        }

        keywords = type_mappings.get(task_type, [])
        return any(keyword in agent_role for keyword in keywords)

    async def _execute_crew(self, crew: 'Crew', tasks: List[Dict], context: Dict) -> str:
        """Execute the crew workflow and return the result."""
        try:
            # Execute the crew
            result = crew.kickoff()

            # Process and format the result
            if hasattr(result, 'raw') and result.raw:
                return str(result.raw)
            elif hasattr(result, 'result') and result.result:
                return str(result.result)
            else:
                return str(result)

        except Exception as e:
            logger.error(f"Error executing crew: {e}")
            raise

    async def _execute_agent_task(self, agent: Agent, user_message: str, system_message: str) -> str:
        """Execute a task with a single agent using their specialized capabilities."""
        try:
            # Determine agent type from role
            agent_role = agent.role.lower()

            # Use specialized agent execution based on type
            # Check for image search requests first (finding images on web)
            if any(word in user_message.lower() for word in ["find", "picture", "image", "photo"]) and any(word in user_message.lower() for word in ["of", "the", "a"]):
                # This is an image search request (finding images), use web search
                return await self._execute_web_search_agent_task(user_message)
            elif "web search" in agent_role or "search" in agent_role:
                return await self._execute_web_search_agent_task(user_message)
            elif "coding" in agent_role or "developer" in agent_role:
                return await self._execute_coding_agent_task(user_message, system_message)
            elif "research" in agent_role or "analyst" in agent_role:
                return await self._execute_research_agent_task(user_message, system_message)
            elif "image" in agent_role or "visual" in agent_role:
                # This is for analyzing uploaded images, not finding images
                return await self._execute_image_analysis_agent_task(user_message, system_message)
            else:
                # Fallback to LLM for general agents
                response = await llm_manager.generate_response(
                    user_message=user_message,
                    system_message=system_message,
                    max_tokens=self._get_agent_max_tokens('general')
                )
                return response

        except Exception as e:
            logger.error(f"Error executing agent task: {e}")
            # Fallback to LLM on error - determine agent type for appropriate token limit
            agent_type = 'general'  # default
            if hasattr(agent, 'role'):
                agent_role = agent.role.lower()
                if "coding" in agent_role or "developer" in agent_role:
                    agent_type = 'coding'
                elif "research" in agent_role or "analyst" in agent_role:
                    agent_type = 'research'
                elif "web search" in agent_role or "search" in agent_role:
                    agent_type = 'web_search'
                elif "image" in agent_role or "visual" in agent_role:
                    agent_type = 'image_analysis'

            try:
                response = await llm_manager.generate_response(
                    user_message=user_message,
                    system_message=system_message,
                    max_tokens=self._get_agent_max_tokens(agent_type)
                )
                return response
            except Exception as fallback_error:
                logger.error(f"Fallback LLM also failed: {fallback_error}")
                raise

    async def _execute_web_search_agent_task(self, user_message: str) -> str:
        """Execute a web search task using the web search agent."""
        try:
            # Import the web search executor
            from agents.web_search_agent import web_search_executor

            # Determine search type based on query
            search_type = "current"  # Default for stock prices, news, etc.
            if any(word in user_message.lower() for word in ["image", "picture", "photo"]):
                search_type = "image"
            elif any(word in user_message.lower() for word in ["price", "stock", "current", "latest", "news"]):
                search_type = "current"
            else:
                search_type = "general"

            # Execute the search
            result = await web_search_executor.execute_search_task(
                query=user_message,
                user_id="agent_user",
                search_type=search_type
            )

            return result

        except Exception as e:
            logger.error(f"Web search agent execution failed: {e}")
            return f"I encountered an error while searching for that information: {str(e)}"

    async def _execute_coding_agent_task(self, user_message: str, system_message: str) -> str:
        """Execute a coding task using the coding agent."""
        try:
            # Import the coding executor
            from agents.coding_agent import coding_executor

            # Execute the coding task
            result = await coding_executor.execute_coding_task(
                query=user_message,
                user_id="agent_user"
            )

            return result

        except Exception as e:
            logger.error(f"Coding agent execution failed: {e}")
            # Fallback to LLM for coding tasks
            response = await llm_manager.generate_response(
                user_message=user_message,
                system_message=system_message,
                max_tokens=self._get_agent_max_tokens('coding')
            )
            return response

    async def _execute_research_agent_task(self, user_message: str, system_message: str) -> str:
        """Execute a research task using the research agent."""
        try:
            # Import the research executor
            from agents.research_agent import research_executor

            # Execute the research task
            result = await research_executor.execute_research_task(
                query=user_message,
                user_id="agent_user"
            )

            return result

        except Exception as e:
            logger.error(f"Research agent execution failed: {e}")
            # Fallback to LLM for research tasks
            response = await llm_manager.generate_response(
                user_message=user_message,
                system_message=system_message,
                max_tokens=self._get_agent_max_tokens('research')
            )
            return response

    async def _execute_image_analysis_agent_task(self, user_message: str, system_message: str) -> str:
        """Execute an image analysis task using the image analysis agent."""
        try:
            # For image analysis, we need an actual image URL
            # This method would be called when there's an image to analyze
            return "Image analysis requires an image to be uploaded. Please upload an image for me to analyze."

        except Exception as e:
            logger.error(f"Image analysis agent execution failed: {e}")
            # Fallback to LLM
            response = await llm_manager.generate_response(
                user_message=user_message,
                system_message=system_message,
                max_tokens=self._get_agent_max_tokens('image_analysis')
            )
            return response

    def _create_enhanced_system_message_for_agent(
        self,
        agent_type: str,
        context: Dict,
        user_message: str,
        user_id: str
    ) -> str:
        """Create an enhanced system message for a specific agent type."""

        # Get user personality
        personality = personality_manager.get_user_personality(user_id)
        base_message = personality.system_message

        # Add context information
        context_info = ""
        if context.get("recent_conversations"):
            context_info += f"\nRecent conversation context:\n{context['recent_conversations'][:500]}...\n"

        if context.get("relevant_memories"):
            context_info += f"\nRelevant memories:\n{context['relevant_memories'][:300]}...\n"

        # Agent-specific instructions
        agent_instructions = {
            'research': """
You are a research specialist. Your role is to:
- Conduct thorough research and analysis
- Provide detailed explanations with evidence
- Compare and contrast different perspectives
- Synthesize information from multiple sources
- Use web search when current information is needed
""",
            'coding': """
You are a coding specialist. Your role is to:
- Write clean, efficient, and well-documented code
- Debug and fix code issues
- Explain programming concepts clearly
- Follow best practices and coding standards
- Provide working code examples

CRITICAL CODE FORMATTING RULES - FOLLOW EXACTLY:
- Use EXACTLY 4 spaces for EVERY indentation level in ALL programming languages
- NEVER use tabs, only spaces
- Every line inside functions, classes, if/for/while blocks MUST be indented
- Count spaces carefully: function body = 4 spaces, nested block = 8 spaces, double-nested = 12 spaces
- Format code in markdown code blocks with language specification
- Example format:
```python
def example_function(param):
    # This line has exactly 4 spaces
    if param:
        # This line has exactly 8 spaces
        for i in range(3):
            # This line has exactly 12 spaces
            print(i)
        return "formatted correctly"
    return "also formatted correctly"
```
""",
            'web_search': """
You are a web search specialist. Your role is to:
- Find current and accurate information online
- Verify information from multiple sources
- Provide specific details like addresses, prices, ratings
- Focus on real-time data and recent updates
- Use Google Search for current information
""",
            'image_analysis': """
You are an image analysis specialist. Your role is to:
- Analyze visual content in detail
- Describe objects, scenes, and activities
- Provide technical details about images
- Offer artistic and compositional insights
- Match commentary to user's personality
""",
            'general': """
RESPONSE GUIDELINES:
- Keep responses SHORT and NATURAL (1-3 sentences for most responses)
- Avoid long explanations unless specifically requested
- Be helpful while maintaining your personality style

MULTILINGUAL SUPPORT:
- ALWAYS respond in the same language the user is using
- Match the user's language naturally and fluently

CODE FORMATTING (when providing code):
- CRITICAL: Use EXACTLY 4 spaces for each indentation level in ALL languages
- NEVER use tabs, always use spaces for indentation
- Format code in markdown code blocks with language specification
- Every line inside functions, classes, if statements, loops MUST be indented with 4 spaces
- Nested blocks get 8 spaces, double-nested get 12 spaces, etc.
- Example:
```python
def example_function(param):
    # This line has 4 spaces
    if param:
        # This line has 8 spaces
        return "formatted correctly"
    return "also formatted correctly"
```

For image requests:
**Images of [subject]:**
🔍 [Google Images](https://www.google.com/search?q=[subject]&tbm=isch)
🔍 [Bing Images](https://www.bing.com/images/search?q=[subject])

IMPORTANT: Maintain your assigned personality style while following these guidelines.
"""
        }

        agent_instruction = agent_instructions.get(agent_type, agent_instructions['general'])

        enhanced_message = f"""{base_message}

{context_info}

{agent_instruction}

Current user request: {user_message}

Remember to maintain your personality while fulfilling your specialist role."""

        return enhanced_message

    def _get_agent_max_tokens(self, agent_type: str) -> int:
        """Get the appropriate max_tokens setting for the agent type."""
        from config import config

        token_limits = {
            'research': config.research_max_tokens,
            'coding': config.coding_max_tokens,
            'general': config.general_max_tokens,
            'web_search': config.general_max_tokens,  # Use general for web search
            'image_analysis': config.general_max_tokens,  # Use general for image analysis
        }

        return token_limits.get(agent_type, config.default_max_tokens)

    async def _fallback_to_llm(self, user_message: str, user_id: str, context: Dict) -> str:
        """Fallback to direct LLM when agents fail."""
        try:
            personality = personality_manager.get_user_personality(user_id)
            enhanced_message = self._create_enhanced_system_message_for_agent(
                'general', context, user_message, user_id
            )

            response = await llm_manager.generate_response(
                user_message=user_message,
                system_message=enhanced_message,
                use_search=True,
                max_tokens=self._get_agent_max_tokens('general')
            )

            return response

        except Exception as e:
            logger.error(f"Error in LLM fallback: {e}")
            return "I'm having trouble processing your request right now. Please try again."

    async def _update_memory(self, user_message: str, response: str, user_id: str, channel_id: str = None):
        """Update memory with conversation - stores ALL conversations for complete recall."""
        try:
            # Update memory with conversation messages (always stored)
            memory_manager.add_conversation_message(user_id, user_message, "user")
            memory_manager.add_conversation_message(user_id, response, "assistant")

            # Add to short-term memory for immediate context
            memory_manager.add_short_term_memory(
                user_id,
                f"User: {user_message}\nAssistant: {response}",
                "conversation",
                importance=0.6
            )

            # ALWAYS store in long-term memory for complete recall
            # This ensures every conversation can be retrieved later
            await memory_manager.add_long_term_memory(
                user_id,
                f"User said: {user_message}",
                memory_type="user_message",
                importance=0.7,
                metadata={"channel_id": channel_id or "unknown", "timestamp": str(datetime.now())}
            )

            # Store bot response for context
            await memory_manager.add_long_term_memory(
                user_id,
                f"Bot responded to '{user_message[:50]}...': {response[:200]}...",
                memory_type="conversation",
                importance=0.6,
                metadata={"channel_id": channel_id or "unknown", "timestamp": str(datetime.now())}
            )

            # Extract and store user preferences or facts with enhanced detection
            await self._extract_and_store_user_info(user_message, response, user_id)

        except Exception as e:
            logger.error(f"Error updating memory: {e}")

    def _is_memory_request(self, user_message: str) -> bool:
        """Check if this is a memory-related request using specific patterns."""
        message_lower = user_message.lower()

        # Explicit memory keywords
        memory_keywords = [
            "remember", "recall", "memory", "told you",
            "do you know about me", "have we talked", "did i tell you",
            "what do you know about me", "tell me about myself"
        ]

        # Personal questions with possessive pronouns (more specific)
        personal_patterns = [
            "what is my", "what's my", "whats my", "who is my", "where is my",
            "what car do i", "what do i drive", "where do i work", "who am i",
            "what have we been talking about", "what information do you have"
        ]

        # Check for explicit memory keywords
        if any(keyword in message_lower for keyword in memory_keywords):
            return True

        # Check for specific personal questions
        if any(pattern in message_lower for pattern in personal_patterns):
            return True

        return False

    def _is_simple_personal_statement(self, user_message: str) -> bool:
        """Check if this is a simple personal statement that should be handled conversationally."""
        message_lower = user_message.lower().strip()

        # Personal statement indicators
        personal_statements = [
            "i am", "i have", "i drive", "i work", "i live", "i like", "i love",
            "i hate", "i own", "i play", "i study", "i enjoy", "my", "mine"
        ]

        # Exclude questions and requests
        if any(word in message_lower for word in ["?", "what", "how", "where", "when", "why", "who", "can you", "please", "help"]):
            return False

        # Exclude search/action requests
        if any(word in message_lower for word in ["search", "find", "look up", "google", "show me", "tell me about"]):
            return False

        # Check if it's a personal statement
        return any(statement in message_lower for statement in personal_statements)

    def _is_simple_greeting_or_casual(self, user_message: str) -> bool:
        """Check if this is a simple greeting or casual message that should go to general agent."""
        message_lower = user_message.lower().strip()

        # Simple greetings and casual messages (multilingual)
        casual_messages = [
            # English
            "hello", "hi", "hey", "good morning", "good afternoon", "good evening",
            "how are you", "how's it going", "what's up", "whats up", "sup",
            "thanks", "thank you", "bye", "goodbye", "see you", "talk later",
            "ok", "okay", "cool", "nice", "awesome", "great", "lol", "haha",
            # Spanish
            "hola", "buenos días", "buenas tardes", "buenas noches", "gracias", "adiós",
            # French
            "bonjour", "bonsoir", "salut", "merci", "au revoir",
            # Italian
            "ciao", "buongiorno", "buonasera", "grazie", "arrivederci",
            # German
            "hallo", "guten tag", "guten morgen", "danke", "auf wiedersehen",
            # Portuguese
            "olá", "bom dia", "boa tarde", "obrigado", "tchau"
        ]

        # Check for exact matches or very short messages
        if message_lower in casual_messages or len(message_lower) <= 3:
            return True

        # Check for greeting patterns (multilingual)
        greeting_patterns = [
            # English
            "good ", "how are", "how's ", "what's up", "whats up",
            # Spanish
            "buenos ", "buenas ", "cómo estás", "como estas",
            # French
            "comment allez", "comment ça va", "ça va",
            # Italian
            "come stai", "come va", "sai parlare", "parli",
            # German
            "wie geht", "guten ", "sprechen sie", "sprichst du",
            # Portuguese
            "como está", "como vai", "você fala"
        ]

        return any(pattern in message_lower for pattern in greeting_patterns)

    async def _smart_route_request(self, user_message: str, user_id: str) -> Dict:
        """Intelligent LLM-based routing - let the LLM decide which agent is best for each request."""

        # Use LLM to intelligently route the request - NO MORE HARDCODED PATTERNS!
        try:
            routing_prompt = f"""You are an intelligent routing system. Analyze this user request and decide which agent should handle it.

USER REQUEST: "{user_message}"

AVAILABLE AGENTS:
- **web_search**: For current/real-time information (weather, stock prices, news, images, live data)
- **coding**: For ANY programming task (write code, debug code, explain code, programming concepts, algorithms)
- **research**: For explanations, concepts, how things work, educational content (NON-programming)
- **general**: For casual conversation, simple math, greetings, personal chat

ROUTING RULES:
- If request mentions: "write", "code", "function", "program", "script", "debug", "python", "javascript", "html", "css", "programming" → choose CODING
- If request asks for current data: "weather", "stock price", "news", "latest" → choose WEB_SEARCH
- If request asks for explanations of non-programming topics → choose RESEARCH
- If request is casual chat, greetings, simple math → choose GENERAL

EXAMPLES:
- "write a python function" → coding
- "what is the weather" → web_search
- "explain quantum physics" → research
- "hello how are you" → general

Respond with ONLY the agent name: web_search, coding, research, or general"""

            response = await llm_manager.generate_response(
                user_message=routing_prompt,
                system_message="You are an expert at understanding user requests and routing them to the appropriate specialist.",
                max_tokens=100
            )

            # Parse the response to get the agent name
            agent_name = response.strip().lower()

            # Validate the agent name
            valid_agents = ['web_search', 'coding', 'research', 'general']
            if agent_name not in valid_agents:
                # Try to extract agent name from response
                for agent in valid_agents:
                    if agent in agent_name:
                        agent_name = agent
                        break
                else:
                    agent_name = 'general'  # fallback

            return {
                'primary_task': agent_name,
                'confidence': 0.9,
                'reasoning': f'LLM intelligently selected {agent_name} agent for this request',
                'method': 'llm_routing'
            }

        except Exception as e:
            logger.error(f"LLM routing failed: {e}")
            # Fallback to general agent
            return {
                'primary_task': 'general',
                'confidence': 0.5,
                'reasoning': 'Fallback due to routing error',
                'method': 'fallback'
            }

    # Old memory handling method completely removed - now using semantic memory system

    # Old hardcoded method removed - now using semantic memory system

    async def _provide_general_summary(self, user_id: str) -> str:
        """Provide a general summary using semantic memory system (no hardcoded patterns)."""
        try:
            # Use the semantic memory system to provide a general summary
            return await self.semantic_memory.handle_memory_request("what do you know about me", user_id)
        except Exception as e:
            logger.error(f"Error providing general summary: {e}")
            return "I'm having trouble accessing what I know about you right now."

    # All hardcoded extraction methods removed - now using semantic memory system

    # Utility methods (still needed)
    async def _extract_and_store_user_info(self, user_message: str, response: str, user_id: str):
        """Store ALL user messages as potentially important information."""
        try:
            # Store EVERY user message as potentially important information
            # No more restrictive phrase matching - remember everything!

            # Skip very short or common messages that aren't informative
            message_lower = user_message.lower().strip()
            skip_phrases = ["hi", "hello", "hey", "ok", "okay", "yes", "no", "thanks", "thank you", "bye"]

            if len(user_message.strip()) > 3 and message_lower not in skip_phrases:
                logger.info(f"Storing user message in memory: '{user_message}'")

                # Store with high importance so it can be recalled
                await memory_manager.add_long_term_memory(
                    user_id,
                    f"User said: {user_message}",
                    memory_type="user_statement",
                    importance=0.8,
                    metadata={"timestamp": str(datetime.now()), "type": "general_info"}
                )

        except Exception as e:
            logger.error(f"Error storing user info: {e}")

    # Removed hardcoded factual pattern extraction - using semantic memory system instead

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



    def _fix_code_formatting(self, response: str) -> str:
        """Fix code formatting and indentation in responses."""
        import re

        if not response:
            return response

        # Find all code blocks (Discord format: ```language\ncode\n```)
        code_block_pattern = r'```([a-zA-Z0-9+#-]+)?\n(.*?)```'

        def fix_code_block(match):
            language = match.group(1) or 'python'
            code = match.group(2)

            if not code.strip():
                return match.group(0)  # Return original if empty

            # Split into lines
            lines = code.split('\n')

            # Remove empty lines at start and end
            while lines and not lines[0].strip():
                lines.pop(0)
            while lines and not lines[-1].strip():
                lines.pop()

            if not lines:
                return match.group(0)  # Return original if no content

            # Fix indentation based on language
            if language.lower() in ['python', 'py']:
                fixed_lines = self._fix_python_indentation(lines)
            elif language.lower() in ['javascript', 'js', 'typescript', 'ts', 'c++', 'cpp', 'c', 'java', 'csharp', 'cs']:
                fixed_lines = self._fix_brace_based_indentation(lines)
            else:
                fixed_lines = self._fix_generic_indentation(lines)

            # Reconstruct code block with Discord-compatible formatting
            fixed_code = '\n'.join(fixed_lines)

            # Ensure proper Discord language specification
            discord_language = language.lower()
            if discord_language == 'c++':
                discord_language = 'cpp'  # Discord prefers 'cpp' over 'c++'
            elif discord_language == 'c#':
                discord_language = 'cs'   # Discord prefers 'cs' over 'c#'

            return f'```{discord_language}\n{fixed_code}\n```'

        # Apply fixes to all code blocks
        fixed_response = re.sub(code_block_pattern, fix_code_block, response, flags=re.DOTALL)

        return fixed_response

    def _fix_python_indentation(self, lines: List[str]) -> List[str]:
        """Fix Python code indentation (4 spaces per level)."""
        fixed_lines = []
        indent_level = 0

        for line in lines:
            stripped = line.strip()
            if not stripped:
                fixed_lines.append('')
                continue

            # Decrease indent for certain keywords
            if stripped.startswith(('except', 'elif', 'else', 'finally')):
                indent_level = max(0, indent_level - 1)
            elif stripped.startswith(('return', 'break', 'continue', 'pass', 'raise')):
                # These don't change indent level but use current level
                pass

            # Apply current indentation
            indented_line = '    ' * indent_level + stripped
            fixed_lines.append(indented_line)

            # Increase indent for lines ending with colon
            if stripped.endswith(':'):
                indent_level += 1

        return fixed_lines

    def _fix_brace_based_indentation(self, lines: List[str]) -> List[str]:
        """Fix brace-based language indentation (C++, JavaScript, Java, C#, etc.) - 4 spaces per level."""
        fixed_lines = []
        indent_level = 0

        for line in lines:
            stripped = line.strip()
            if not stripped:
                fixed_lines.append('')
                continue

            # Handle closing braces - decrease indent before applying
            if stripped.startswith(('}', ')', ']')):
                indent_level = max(0, indent_level - 1)

            # Handle special cases like 'else', 'else if', 'catch', etc.
            if stripped.startswith(('else', 'catch', 'finally')) and not stripped.endswith(('{', ';')):
                # These are at the same level as the previous block
                pass

            # Apply current indentation (4 spaces per level for Discord compatibility)
            indented_line = '    ' * indent_level + stripped
            fixed_lines.append(indented_line)

            # Handle opening braces - increase indent after applying
            if stripped.endswith(('{', '(', '[')):
                indent_level += 1

            # Handle single-line statements that should be indented (like for/if without braces)
            if any(stripped.startswith(keyword) for keyword in ['for', 'while', 'if', 'else if']) and not stripped.endswith('{'):
                # Check if next line should be indented (single statement)
                # This will be handled by the next iteration
                pass

        return fixed_lines

    def _fix_generic_indentation(self, lines: List[str]) -> List[str]:
        """Fix generic code indentation (2 spaces per level)."""
        fixed_lines = []
        indent_level = 0

        for line in lines:
            stripped = line.strip()
            if not stripped:
                fixed_lines.append('')
                continue

            # Simple heuristic: decrease for closing brackets/braces
            if stripped.startswith(('}', ')', ']')):
                indent_level = max(0, indent_level - 1)

            # Apply current indentation
            indented_line = '  ' * indent_level + stripped
            fixed_lines.append(indented_line)

            # Increase for opening brackets/braces or colons
            if stripped.endswith(('{', '(', '[', ':')):
                indent_level += 1

        return fixed_lines

    def _create_enhanced_system_message(self, base_system_message: str, context: Dict, user_message: str) -> str:
        """Create an enhanced system message with context and capabilities."""

        # Add context information
        context_info = ""
        if context.get("recent_conversations"):
            context_info += f"\nRecent conversation context:\n{context['recent_conversations'][:500]}...\n"

        if context.get("relevant_memories"):
            context_info += f"\nRelevant memories:\n{context['relevant_memories'][:300]}...\n"

        # Create comprehensive system message
        enhanced_message = f"""{base_system_message}

{context_info}

You are an advanced AI assistant with Google Search capabilities. You have access to:
- Real-time information through Google Search
- Your extensive knowledge base for general topics
- Code writing, debugging, and technical assistance
- Research and detailed explanations
- General conversation and assistance
- Memory of past conversations

Instructions:
- For questions requiring current/real-time information (weather, restaurant listings, stock prices, news, business hours, etc.), use Google Search to provide accurate, up-to-date responses
- For general knowledge, technical explanations, coding help, and educational topics, use your built-in knowledge
- You decide when to search - trust your judgment about what needs current information
- When you search, provide specific, helpful information with details like addresses, phone numbers, ratings, etc.
- For coding requests, provide working code with explanations
- Be helpful, accurate, and engaging
- Remember context from our conversation

User's current request: {user_message}"""

        return enhanced_message

    # Removed manual detection logic - Gemini decides when to search!

    # Removed manual search handling - Gemini handles everything!



    # Utility methods (still needed)
    async def _extract_and_store_user_info(self, user_message: str, response: str, user_id: str):
        """Extract and store user preferences or important facts."""
        try:
            # Simple extraction of user preferences
            if any(phrase in user_message.lower() for phrase in ["i like", "i prefer", "my favorite", "i am", "i work"]):
                await memory_manager.add_long_term_memory(
                    user_id,
                    f"User preference/info: {user_message}",
                    memory_type="user_info",
                    importance=0.7
                )
        except Exception as e:
            logger.error(f"Error extracting user info: {e}")

    def _clean_final_response(self, response: str) -> str:
        """Clean the final response to remove any unwanted elements."""
        if not response:
            return response

        import re

        # Remove thinking tags and their content
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL | re.IGNORECASE)
        response = re.sub(r'<thinking>.*?</thinking>', '', response, flags=re.DOTALL | re.IGNORECASE)
        response = re.sub(r'<thought>.*?</thought>', '', response, flags=re.DOTALL | re.IGNORECASE)

        # Remove tool code blocks and attempts
        response = re.sub(r'```tool_code.*?```', '', response, flags=re.DOTALL | re.IGNORECASE)
        response = re.sub(r'```python.*?print\(goog.*?```', '', response, flags=re.DOTALL | re.IGNORECASE)
        response = re.sub(r'print\(goog[^)]*\)', '', response, flags=re.IGNORECASE)

        # Remove excessive whitespace
        response = re.sub(r'\n\s*\n\s*\n', '\n\n', response)
        response = response.strip()

        return response

# Legacy CrewOrchestrator class for backward compatibility
class CrewOrchestrator:
    """Legacy orchestrator - redirects to new multi-agent system."""

    def __init__(self):
        self.multi_agent_orchestrator = MultiAgentCrewOrchestrator()

    async def process_request(self, user_message: str, user_id: str, channel_id: str = None) -> str:
        """Legacy method - redirects to new multi-agent system."""
        return await self.multi_agent_orchestrator.process_request(user_message, user_id, channel_id)

    def _clean_final_response(self, response: str) -> str:
        """Legacy method - redirects to new system."""
        return self.multi_agent_orchestrator._clean_final_response(response)

# Global orchestrator instances
multi_agent_orchestrator = MultiAgentCrewOrchestrator()
crew_orchestrator = CrewOrchestrator()  # For backward compatibility
