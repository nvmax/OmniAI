"""
Coding tools for code generation, execution, and file management.
Provides safe code execution and file handling capabilities.
"""

import os
import subprocess
import tempfile
import logging
from typing import Optional, Dict, Any
from pathlib import Path
from crewai.tools.base_tool import BaseTool

logger = logging.getLogger(__name__)

class CodeExecutionTool(BaseTool):
    """Tool for safely executing code in a controlled environment."""
    
    name: str = "code_execution"
    description: str = (
        "Execute Python code safely in a temporary environment. "
        "Provide Python code to execute and get the output. "
        "Use this for testing code snippets, calculations, or demonstrations."
    )
    
    def __init__(self):
        super().__init__()
        self._timeout = 30  # 30 second timeout
        self._allowed_imports = {
            'math', 'random', 'datetime', 'json', 'os', 'sys', 'time',
            'collections', 'itertools', 'functools', 're', 'string',
            'urllib.parse', 'base64', 'hashlib', 'uuid'
        }
    
    def _is_safe_code(self, code: str) -> tuple[bool, str]:
        """Check if code is safe to execute."""
        dangerous_patterns = [
            'import subprocess', 'import os', '__import__', 'eval(', 'exec(',
            'open(', 'file(', 'input(', 'raw_input(', 'compile(',
            'globals(', 'locals(', 'vars(', 'dir(', 'getattr(', 'setattr(',
            'delattr(', 'hasattr(', '__builtins__', '__globals__'
        ]
        
        code_lower = code.lower()
        for pattern in dangerous_patterns:
            if pattern in code_lower:
                return False, f"Potentially unsafe pattern detected: {pattern}"
        
        return True, "Code appears safe"
    
    def _run(self, code: str) -> str:
        """Execute Python code safely."""
        try:
            # Check if code is safe
            is_safe, safety_message = self._is_safe_code(code)
            if not is_safe:
                return f"Code execution blocked: {safety_message}"
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            try:
                # Execute code with timeout
                result = subprocess.run(
                    ['python', temp_file],
                    capture_output=True,
                    text=True,
                    timeout=self._timeout,
                    cwd=tempfile.gettempdir()
                )
                
                output = ""
                if result.stdout:
                    output += f"Output:\n{result.stdout}\n"
                if result.stderr:
                    output += f"Errors:\n{result.stderr}\n"
                if result.returncode != 0:
                    output += f"Exit code: {result.returncode}\n"
                
                return output if output else "Code executed successfully with no output."
                
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_file)
                except:
                    pass
                    
        except subprocess.TimeoutExpired:
            return f"Code execution timed out after {self._timeout} seconds."
        except Exception as e:
            logger.error(f"Error executing code: {e}")
            return f"Error executing code: {str(e)}"

class FileReaderTool(BaseTool):
    """Tool for reading and analyzing files safely."""
    
    name: str = "file_reader"
    description: str = (
        "Read and analyze text files. Provide a file path to read its contents. "
        "Supports common text file formats and provides basic analysis."
    )
    
    def __init__(self):
        super().__init__()
        self._max_file_size = 1024 * 1024  # 1MB limit
        self._allowed_extensions = {
            '.txt', '.py', '.js', '.html', '.css', '.json', '.xml',
            '.md', '.yml', '.yaml', '.ini', '.cfg', '.conf'
        }
    
    def _run(self, file_path: str) -> str:
        """Read and analyze a file."""
        try:
            path = Path(file_path)
            
            # Security checks
            if not path.exists():
                return f"File not found: {file_path}"
            
            if not path.is_file():
                return f"Path is not a file: {file_path}"
            
            if path.suffix.lower() not in self._allowed_extensions:
                return f"File type not allowed: {path.suffix}"

            if path.stat().st_size > self._max_file_size:
                return f"File too large (max {self._max_file_size} bytes): {path.stat().st_size} bytes"
            
            # Read file content
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except UnicodeDecodeError:
                with open(path, 'r', encoding='latin-1') as f:
                    content = f.read()
            
            # Basic analysis
            lines = content.split('\n')
            analysis = {
                'file_name': path.name,
                'file_size': path.stat().st_size,
                'line_count': len(lines),
                'character_count': len(content),
                'file_type': path.suffix
            }
            
            # Truncate content if too long
            if len(content) > 2000:
                content = content[:2000] + "\n... [Content truncated]"
            
            return f"""File Analysis:
Name: {analysis['file_name']}
Type: {analysis['file_type']}
Size: {analysis['file_size']} bytes
Lines: {analysis['line_count']}
Characters: {analysis['character_count']}

Content:
{content}"""
            
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return f"Error reading file: {str(e)}"

class CodeGenerationTool(BaseTool):
    """Tool for generating code snippets and templates."""
    
    name: str = "code_generation"
    description: str = (
        "Generate code snippets, templates, and examples for various programming tasks. "
        "Specify the programming language and describe what you want to create."
    )
    
    def __init__(self):
        super().__init__()
        self._templates = {
            'python': {
                'function': '''def {name}({params}):
    """
    {description}
    """
    # TODO: Implement function logic
    pass''',
                'class': '''class {name}:
    """
    {description}
    """
    
    def __init__(self{params}):
        # TODO: Initialize class attributes
        pass''',
                'script': '''#!/usr/bin/env python3
"""
{description}
"""

def main():
    # TODO: Implement main logic
    pass

if __name__ == "__main__":
    main()'''
            },
            'javascript': {
                'function': '''function {name}({params}) {{
    // {description}
    // TODO: Implement function logic
}}''',
                'class': '''class {name} {{
    constructor({params}) {{
        // {description}
        // TODO: Initialize class properties
    }}
}}''',
                'async': '''async function {name}({params}) {{
    try {{
        // {description}
        // TODO: Implement async logic
    }} catch (error) {{
        console.error('Error in {name}:', error);
        throw error;
    }}
}}'''
            }
        }
    
    def _run(self, request: str) -> str:
        """Generate code based on the request."""
        try:
            request_lower = request.lower()
            
            # Simple pattern matching for code generation
            if 'python' in request_lower and 'function' in request_lower:
                return self._templates['python']['function'].format(
                    name='example_function',
                    params='param1, param2',
                    description='Example function description'
                )
            elif 'python' in request_lower and 'class' in request_lower:
                return self._templates['python']['class'].format(
                    name='ExampleClass',
                    params=', param1, param2',
                    description='Example class description'
                )
            elif 'javascript' in request_lower and 'function' in request_lower:
                return self._templates['javascript']['function'].format(
                    name='exampleFunction',
                    params='param1, param2',
                    description='Example function description'
                )
            else:
                return f"""I can help generate code templates. Here are some examples:

Python Function:
{self._templates['python']['function'].format(
    name='my_function',
    params='arg1, arg2',
    description='Function description'
)}

Python Class:
{self._templates['python']['class'].format(
    name='MyClass',
    params=', arg1, arg2',
    description='Class description'
)}

Please specify the programming language and type of code you need."""
                
        except Exception as e:
            logger.error(f"Error generating code: {e}")
            return f"Error generating code: {str(e)}"
