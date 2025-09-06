"""
Coding agent for code generation, debugging, and programming assistance.
Specializes in software development tasks and technical problem-solving.
"""

import logging
from typing import List, Optional, Dict, Any
from crewai import Agent

try:
    from .base_agent import AgentFactory
    from ..tools.coding_tool import CodeExecutionTool, FileReaderTool, CodeGenerationTool
    from ..tools.vector_db_tool import VectorDBRetrievalTool, MemoryStorageTool
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from agents.base_agent import AgentFactory
    from tools.coding_tool import CodeExecutionTool, FileReaderTool, CodeGenerationTool
    from tools.vector_db_tool import VectorDBRetrievalTool, MemoryStorageTool

logger = logging.getLogger(__name__)

class CodingAgentManager:
    """Manager for coding agent operations."""
    
    def __init__(self):
        self.code_execution = CodeExecutionTool()
        self.file_reader = FileReaderTool()
        self.code_generation = CodeGenerationTool()
        self.memory_search = VectorDBRetrievalTool()
        self.memory_storage = MemoryStorageTool()
    
    def get_coding_tools(self) -> List:
        """Get all tools available to the coding agent."""
        return [
            self.code_execution,
            self.file_reader,
            self.code_generation,
            self.memory_search,
            self.memory_storage
        ]
    
    def create_coding_agent(self, user_id: str = "default") -> Agent:
        """Create a coding agent with all necessary tools."""
        tools = self.get_coding_tools()
        return AgentFactory.create_coding_agent(user_id, tools)

    def create_agent(self, user_id: str = "default") -> Agent:
        """Create a coding agent (alias for create_coding_agent)."""
        return self.create_coding_agent(user_id)
    
    def analyze_coding_request(self, request: str) -> Dict[str, Any]:
        """Analyze a coding request to determine the best approach."""
        request_lower = request.lower()
        
        analysis = {
            "task_type": "general",
            "language": "unknown",
            "needs_execution": False,
            "needs_file_reading": False,
            "needs_generation": False,
            "complexity": "medium",
            "suggested_approach": []
        }
        
        # Determine programming language
        languages = {
            "python": ["python", "py", "django", "flask", "pandas", "numpy"],
            "javascript": ["javascript", "js", "node", "react", "vue", "angular"],
            "html": ["html", "web", "webpage", "website"],
            "css": ["css", "style", "styling", "bootstrap"],
            "sql": ["sql", "database", "query", "select", "insert"],
            "json": ["json", "api", "data"]
        }
        
        for lang, keywords in languages.items():
            if any(keyword in request_lower for keyword in keywords):
                analysis["language"] = lang
                break
        
        # Determine task type
        if any(word in request_lower for word in ["debug", "fix", "error", "bug", "problem"]):
            analysis["task_type"] = "debugging"
            analysis["needs_execution"] = True
        elif any(word in request_lower for word in ["create", "write", "generate", "make", "build"]):
            analysis["task_type"] = "generation"
            analysis["needs_generation"] = True
        elif any(word in request_lower for word in ["explain", "understand", "how does", "what does"]):
            analysis["task_type"] = "explanation"
        elif any(word in request_lower for word in ["test", "run", "execute", "check"]):
            analysis["task_type"] = "testing"
            analysis["needs_execution"] = True
        elif any(word in request_lower for word in ["read", "analyze", "review", "file"]):
            analysis["task_type"] = "analysis"
            analysis["needs_file_reading"] = True
        
        # Determine if file reading is needed
        if any(word in request_lower for word in ["file", "script", "code in", "analyze this"]):
            analysis["needs_file_reading"] = True
        
        # Build suggested approach
        if analysis["task_type"] == "debugging":
            analysis["suggested_approach"] = [
                "1. Analyze the code for common issues",
                "2. Check syntax and logic errors",
                "3. Test the code if safe to execute",
                "4. Provide corrected version with explanations"
            ]
        elif analysis["task_type"] == "generation":
            analysis["suggested_approach"] = [
                "1. Understand requirements clearly",
                "2. Design the code structure",
                "3. Generate clean, documented code",
                "4. Provide usage examples"
            ]
        elif analysis["task_type"] == "explanation":
            analysis["suggested_approach"] = [
                "1. Break down the code into components",
                "2. Explain each part's functionality",
                "3. Highlight key concepts and patterns",
                "4. Provide context and best practices"
            ]
        
        return analysis

class CodingTaskExecutor:
    """Executes coding tasks using the coding agent."""
    
    def __init__(self):
        self.agent_manager = CodingAgentManager()
    
    async def execute_coding_task(self, request: str, user_id: str = "default") -> str:
        """Execute a coding task and return comprehensive results."""
        try:
            # Analyze the request
            analysis = self.agent_manager.analyze_coding_request(request)
            
            # Create coding agent
            coding_agent = self.agent_manager.create_coding_agent(user_id)
            
            # Check memory for relevant coding patterns or solutions
            memory_tool = self.agent_manager.memory_search
            memory_results = memory_tool._run(request, user_id)
            
            results = []
            
            # Add memory context if relevant
            if "No relevant information found" not in memory_results:
                results.append(f"Relevant Past Context:\n{memory_results}")
            
            # Execute based on task type
            if analysis["task_type"] == "generation":
                generation_result = await self._handle_code_generation(request, analysis)
                results.append(generation_result)
            
            elif analysis["task_type"] == "debugging":
                debug_result = await self._handle_debugging(request, analysis)
                results.append(debug_result)
            
            elif analysis["task_type"] == "explanation":
                explanation_result = await self._handle_code_explanation(request, analysis)
                results.append(explanation_result)
            
            elif analysis["task_type"] == "testing":
                test_result = await self._handle_code_testing(request, analysis)
                results.append(test_result)
            
            elif analysis["task_type"] == "analysis":
                analysis_result = await self._handle_code_analysis(request, analysis)
                results.append(analysis_result)
            
            else:
                # General coding assistance
                general_result = await self._handle_general_coding(request, analysis)
                results.append(general_result)
            
            # Compile final response
            final_response = "\n\n".join(results)
            
            # Store important coding solutions in memory
            if analysis["task_type"] in ["generation", "debugging"] and len(final_response) > 100:
                storage_tool = self.agent_manager.memory_storage
                storage_tool._run(
                    f"Coding solution for '{request}': {final_response[:500]}...",
                    user_id,
                    "coding"
                )
            
            return final_response
            
        except Exception as e:
            logger.error(f"Error executing coding task: {e}")
            return f"I encountered an error while working on your coding request: {str(e)}"
    
    async def _handle_code_generation(self, request: str, analysis: Dict) -> str:
        """Handle code generation requests."""
        generation_tool = self.agent_manager.code_generation
        generated_code = generation_tool._run(request)
        
        return f"Generated Code ({analysis['language']}):\n\n{generated_code}\n\nApproach:\n" + "\n".join(analysis["suggested_approach"])
    
    async def _handle_debugging(self, request: str, analysis: Dict) -> str:
        """Handle debugging requests."""
        result_parts = [
            f"Debugging Analysis ({analysis['language']}):",
            "",
            "Common issues to check:",
            "- Syntax errors (missing colons, brackets, quotes)",
            "- Indentation problems",
            "- Variable naming and scope issues",
            "- Logic errors in conditions and loops",
            "- Import/dependency issues",
            "",
            "Debugging approach:"
        ]
        
        result_parts.extend(analysis["suggested_approach"])
        
        if "code" in request.lower():
            result_parts.append("\nIf you provide the specific code, I can help identify and fix the issues.")
        
        return "\n".join(result_parts)
    
    async def _handle_code_explanation(self, request: str, analysis: Dict) -> str:
        """Handle code explanation requests."""
        result_parts = [
            f"Code Explanation Approach ({analysis['language']}):",
            "",
            "I'll help explain the code by:"
        ]
        
        result_parts.extend(analysis["suggested_approach"])
        
        if "code" in request.lower():
            result_parts.append("\nPlease share the specific code you'd like me to explain.")
        
        return "\n".join(result_parts)
    
    async def _handle_code_testing(self, request: str, analysis: Dict) -> str:
        """Handle code testing requests."""
        if analysis["needs_execution"]:
            execution_tool = self.agent_manager.code_execution
            
            # This would need actual code to test
            return f"Code Testing ({analysis['language']}):\n\nI can help test your code safely. Please provide the code you'd like me to test, and I'll:\n\n1. Review it for safety\n2. Execute it in a controlled environment\n3. Report the results\n4. Suggest improvements if needed"
        else:
            return "Please provide the code you'd like me to test."
    
    async def _handle_code_analysis(self, request: str, analysis: Dict) -> str:
        """Handle code analysis requests."""
        if analysis["needs_file_reading"]:
            return f"Code Analysis ({analysis['language']}):\n\nI can analyze code files for you. Please provide:\n\n1. The file path (if it's a local file)\n2. Or paste the code directly\n\nI'll analyze:\n- Code structure and organization\n- Potential improvements\n- Best practices compliance\n- Performance considerations"
        else:
            return "Please provide the code or file path you'd like me to analyze."
    
    async def _handle_general_coding(self, request: str, analysis: Dict) -> str:
        """Handle general coding assistance."""
        return f"Coding Assistance ({analysis['language']}):\n\nI can help you with:\n\n- Writing new code\n- Debugging existing code\n- Code review and optimization\n- Explaining programming concepts\n- Best practices and patterns\n\nWhat specific coding task would you like help with?"

# Global coding manager instance
coding_manager = CodingAgentManager()
coding_executor = CodingTaskExecutor()
