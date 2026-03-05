import json
import logging
import os
import sys
from abc import ABC, abstractmethod
from typing import Optional

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """Abstract base class for LLM providers"""

    @abstractmethod
    def generate_code(self, prompt: str, system_prompt: str) -> str:
        """Generate Python code from a natural language prompt"""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI GPT-4 provider for code generation"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        try:
            import openai

            self.client = openai.OpenAI(api_key=api_key)
        except ImportError:
            logger.error("openai package not installed. Install with: pip install openai")
            sys.exit(1)

    def generate_code(self, prompt: str, system_prompt: str) -> str:
        """Generate Python code using OpenAI's API"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=3500,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise RuntimeError(f"Failed to generate code with OpenAI: {e}")


class ClaudeProvider(LLMProvider):
    """Anthropic Claude provider for code generation"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        try:
            import anthropic

            self.client = anthropic.Anthropic(api_key=api_key)
        except ImportError:
            logger.error("anthropic package not installed. Install with: pip install anthropic")
            sys.exit(1)

    def generate_code(self, prompt: str, system_prompt: str) -> str:
        """Generate Python code using Claude's API"""
        try:
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=3500,
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Claude API error: {e}")
            raise RuntimeError(f"Failed to generate code with Claude: {e}")


class AgentCodeGenerator:
    """Generates and manages agent code execution"""

    # System prompt that constrains the agent's behavior
    SYSTEM_PROMPT = """You are a FaaSr agent that can interact with S3 storage to process user data.

You have access to the following safe FaaSr functions:
- faasr_put_file(local_file, remote_file, local_folder=".", remote_folder="."): Upload files to S3
- faasr_get_file(local_file, remote_file, local_folder=".", remote_folder="."): Download files from S3
- faasr_get_folder_list(prefix=""): List files in S3 by prefix
- faasr_log(message): Log a message
- faasr_invocation_id(): Get the current invocation ID
- faasr_rank(): Get current rank and max rank

IMPORTANT CONSTRAINTS:
1. You MUST NOT attempt to modify, overwrite, or delete existing files
2. You MUST use descriptive file names and avoid naming conflicts
3. You MUST limit your operations to reasonable numbers (max 40 S3 requests)
4. You MUST NOT attempt to access or expose any secrets or credentials
5. You MUST NOT make HTTP requests to external APIs (unless already in the environment)
6. You MUST write any generated files to /tmp/ before uploading to S3
7. You MUST handle errors gracefully with try-except blocks
8. You SHOULD explore available data before deciding how to process it
9. You SHOULD make intelligent decisions based on what you discover

Your task is to write Python code that accomplishes the user's request. The code will be executed
in a sandboxed environment with access only to the functions listed above. Generate ONLY the Python
code without any markdown formatting, explanations, or code blocks. The code should be complete and
ready to execute directly.

IMPORTANT: Write adaptive code that explores and makes decisions:
- First explore what files exist using faasr_get_folder_list
- Inspect file contents to understand the data structure
- Make decisions based on what you find (if/else, different processing paths)
- Handle missing files gracefully and adapt your approach
- Think like a detective - gather clues, then act based on what you discover
- Use faasr_log to document what you're finding and what decisions you're making

Example adaptive pattern:
```
files = faasr_get_folder_list()
if "config.json" in files:
    # Load config and use it to guide processing
elif "settings.ini" in files:
    # Different approach for ini files
else:
    # Fallback to scanning for data files
```

Focus on:
- Exploration first (what's available?)
- Inspection second (what's in the files?)
- Adaptive processing based on findings
- Logging discoveries and decisions"""

    def __init__(self, api_key: str, provider: str = "openai"):
        """
        Initialize the code generator

        Arguments:
            api_key: API key for the LLM provider
            provider: "openai" or "claude"
        """
        self.provider_name = provider.lower()
        
        if self.provider_name == "openai":
            self.llm = OpenAIProvider(api_key)
        elif self.provider_name == "claude":
            self.llm = ClaudeProvider(api_key)
        else:
            raise ValueError(f"Unknown provider: {provider}. Use 'openai' or 'claude'")

    def generate_code(self, prompt: str) -> str:
        """
        Generate Python code from a natural language prompt

        Arguments:
            prompt: Natural language description of what the agent should do

        Returns:
            Python code as string
        """
        logger.info(f"Generating code using {self.provider_name} for prompt: {prompt[:100]}...")
        
        code = self.llm.generate_code(prompt, self.SYSTEM_PROMPT)
        
        # Clean up markdown formatting if present
        code = self._clean_code(code)
        
        logger.debug(f"Generated code:\n{code}")
        return code

    def generate_code_with_context(self, prompt: str, exploration_data: dict) -> str:
        """
        Generate Python code with S3 exploration context
        
        Arguments:
            prompt: Natural language description
            exploration_data: S3 file structure information
            
        Returns:
            Python code as string
        """
        # Enhance prompt with context
        context_info = ""
        if exploration_data.get("files"):
            context_info = f"\n\nAvailable S3 files ({exploration_data['file_count']} total):\n"
            if exploration_data['file_count'] <= 20:
                context_info += "\n".join(f"- {f}" for f in exploration_data['files'])
            else:
                context_info += "\n".join(f"- {f}" for f in exploration_data['files'][:10])
                context_info += f"\n... and {exploration_data['file_count'] - 10} more files"
                
            if exploration_data.get('folders'):
                context_info += f"\n\nFolders: {', '.join(exploration_data['folders'])}"
        
        enhanced_prompt = prompt + context_info
        
        # Enhanced system prompt for adaptive behavior
        adaptive_system_prompt = self.SYSTEM_PROMPT + """

ADAPTIVE EXPLORATION:
You can explore data and then change your approach by:
1. First downloading and examining sample files
2. Based on what you find, you can call _regenerate_approach(new_prompt, discovered_data) to get a new strategy
3. This allows you to adapt based on actual data structure, formats, or content

Example adaptive pattern:
```python
# Initial exploration
sample_file = faasr_get_folder_list()[0]  
faasr_get_file(f"/tmp/{sample_file}", sample_file)

with open(f"/tmp/{sample_file}") as f:
    content = f.read()
    
# Discover data format
if "CSV" in content[:100]:
    # Regenerate approach for CSV processing
    _regenerate_approach(
        "Process all files as CSV with these columns: ...",
        {"format": "csv", "sample": content[:500]}
    )
```"""
        
        logger.info(f"Generating context-aware code for: {prompt[:100]}...")
        code = self.llm.generate_code(enhanced_prompt, adaptive_system_prompt)
        code = self._clean_code(code)
        return code

    @staticmethod
    def _clean_code(code: str) -> str:
        """Remove markdown formatting from generated code"""
        # Remove ```python and ``` markers
        if code.startswith("```python"):
            code = code[9:]
        elif code.startswith("```"):
            code = code[3:]
        
        if code.endswith("```"):
            code = code[:-3]
        
        return code.strip()

    @staticmethod
    def validate_code_safety(code: str) -> bool:
        """
        Perform basic safety checks on generated code

        Arguments:
            code: Python code to validate

        Returns:
            True if code passes safety checks, False otherwise
        """
        dangerous_patterns = [
            "__import__",
            "eval",
            "exec",
            "compile",
            "open(",  # Direct file operations
            "os.system",
            "subprocess",
            "import requests",  # HTTP requests
            "socket",
            "__file__",
            "globals()",
            "locals()",
            "vars(",
        ]

        code_lower = code.lower()
        for pattern in dangerous_patterns:
            if pattern.lower() in code_lower:
                logger.warning(f"Dangerous pattern detected in generated code: {pattern}")
                return False

        return True


def get_agent_provider() -> Optional[str]:
    """
    Detect the appropriate LLM provider based on available API keys

    Returns:
        "openai", "claude", or None if no keys found
    """
    if os.getenv("AGENT_KEY"):
        # Check Claude keys first (they start with 'sk-ant-')
        key = os.getenv("AGENT_KEY")
        if key.startswith("sk-ant-") or "claude" in key.lower():
            return "claude"
        # Check if it's OpenAI format (starts with 'sk-')
        elif key.startswith("sk-"):
            return "openai"
    
    return None


def get_agent_api_key() -> str:
    """
    Get the agent API key from environment

    Returns:
        API key string

    Raises:
        RuntimeError if AGENT_KEY not found
    """
    api_key = os.getenv("AGENT_KEY")
    if not api_key:
        raise RuntimeError(
            "AGENT_KEY not found in environment. Please set AGENT_KEY secret."
        )
    return api_key


class AdaptiveAgentExecutor:
    """Manages adaptive agent execution with one-time regeneration"""

    def __init__(self, faasr, prompt, action_name, api_key, provider):
        self.faasr = faasr
        self.prompt = prompt
        self.action_name = action_name
        self.generator = AgentCodeGenerator(api_key, provider)
        self.context = AgentContextManager(faasr)
        self.regeneration_used = False
        self.max_regenerations = 1

    def execute_with_regeneration(self, code: str) -> tuple[bool, dict]:
        """Execute code and return exploration data for potential regeneration"""
        from FaaSr_py.client.agent_func_entry import _prepare_agent_namespace
        
        logger.info("Executing agent code")
        env_backup = self.context.sanitize_environment()
        exploration_data = {}
        
        try:
            agent_namespace = _prepare_agent_namespace(self.faasr, self.context)
            
            # Callback for agent to request regeneration
            def request_regeneration(discovery_info: dict):
                nonlocal exploration_data
                exploration_data = discovery_info
                exploration_data["needs_regeneration"] = True
            
            agent_namespace["_request_regeneration"] = request_regeneration
            
            exec(code, agent_namespace)
            result = agent_namespace.get("result", True)
            return True, {"result": result, "exploration": exploration_data}
        except Exception as e:
            logger.error(f"Execution failed: {e}")
            return False, {"error": str(e), "exploration": exploration_data}
        finally:
            self.context.restore_environment(env_backup)

    def execute(self) -> bool:
        """Execute adaptive agent with optional regeneration"""
        logger.info(f"Starting adaptive agent: {self.action_name}")
        
        # Phase 1: Explore and initial execution
        code = self.generator.generate_code(self.prompt)
        if not self.generator.validate_code_safety(code):
            logger.error("Generated code failed safety validation")
            return False
        
        success, execution_result = self.execute_with_regeneration(code)
        if not success:
            logger.error(f"Execution failed: {execution_result}")
            return False
        
        # Phase 2: Check if regeneration is needed (only once)
        exploration = execution_result.get("exploration", {})
        if exploration.get("needs_regeneration") and not self.regeneration_used:
            logger.info("Regenerating code based on discoveries")
            self.regeneration_used = True
            
            discovery_prompt = f"{self.prompt}\n\nBased on exploration, I discovered:\n{json.dumps(exploration, indent=2)}"
            regenerated_code = self.generator.generate_code(discovery_prompt)
            
            if not self.generator.validate_code_safety(regenerated_code):
                logger.error("Regenerated code failed safety validation")
                return False
            
            success, final_result = self.execute_with_regeneration(regenerated_code)
            if not success:
                logger.error(f"Regenerated execution failed: {final_result}")
                return False
            
            logger.info("Regenerated execution completed")
        elif exploration.get("needs_regeneration"):
            logger.warning("Regeneration requested but limit already reached")
        
        logger.info("Adaptive agent execution completed")
        return True
