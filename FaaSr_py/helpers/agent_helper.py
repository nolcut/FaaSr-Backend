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
                model="claude-haiku-4-5-20251001",
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

CRITICAL OUTPUT RULES:
- Generate ONLY pure Python code - no markdown, no triple backticks, no ```python tags
- Do NOT include any explanatory text before or after the code
- Do NOT wrap code in any formatting or code blocks
- The output must be directly executable Python code
- Start immediately with import statements or code, no pretext

CRITICAL RUNTIME RULES:
- DO NOT import or reference any 'faasr' module (there is no module to import)
- DO NOT modify sys.path or use hardcoded paths like /opt/faas/faasr
- Use ONLY the provided functions (faasr_put_file, faasr_get_file, faasr_get_folder_list, faasr_log,
  faasr_invocation_id, faasr_rank) directly without imports

You have access to the following safe FaaSr functions:
- faasr_put_file(local_file, remote_file, server_name="", local_folder=".", remote_folder="."): Upload files to S3
- faasr_get_file(local_file, remote_file, server_name="", local_folder=".", remote_folder="."): Download files from S3
- faasr_get_folder_list(server_name="", prefix=""): List files in S3 by prefix
- faasr_log(message): Log a message
- faasr_invocation_id(): Get the current invocation ID
- faasr_rank(): Get current rank and max rank

These functions are injected into the runtime from agent_stubs.py. Do NOT import any module to access them.

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
10. You MUST upload the final response artifact to S3 using faasr_put_file
11. You MUST NOT pass None to any faasr_* API call (use empty strings or omit optional args)
12. For optional arguments, prefer keyword arguments. Example: faasr_get_folder_list(prefix="my/path/")

Your task is to write Python code that accomplishes the user's request. The code will be executed
in a sandboxed environment with access only to the functions listed above. 

IMPORTANT: Write adaptive code that explores and makes decisions:
- Inspect file contents to understand the data structure
- Make decisions based on what you find (if/else, different processing paths)
- Handle missing files gracefully and adapt your approach
- Think like a detective - gather clues, then act based on what you discover
- Use faasr_log to document what you're finding and what decisions you're making

STRICT DATA-DRIVEN BEHAVIOR:
- Never fabricate files, results, or content
- If information isn't found, log what you checked and exit gracefully
- Your output must reflect the actual data discovered during exploration

Focus on:
- Exploration first (what's available?)
- Inspection second (what's in the files?)
- Adaptive processing based on findings
- Logging discoveries and decisions"""

    def __init__(self, api_key: str, provider: str):
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

    def generate_text(self, prompt: str, system_prompt: str) -> str:
        """
        Generate plain text from a prompt with a custom system prompt

        Arguments:
            prompt: Natural language prompt
            system_prompt: Custom system instruction

        Returns:
            Raw text as string
        """
        logger.info("Generating text response with custom system prompt")
        return self.llm.generate_code(prompt, system_prompt)

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

        if exploration_data.get("file_previews"):
            context_info += "\n\nFile previews (truncated):\n"
            for file_name, preview in exploration_data["file_previews"].items():
                context_info += f"\n--- {file_name} ---\n{preview}\n"
        
        enhanced_prompt = prompt + context_info
        
        # Enhanced system prompt for adaptive behavior
        adaptive_system_prompt = self.SYSTEM_PROMPT + """

ADAPTIVE EXPLORATION AND INSIGHTS:
1. First download and examine actual data files - don't make assumptions
2. Analyze real content, structure, and patterns in the data
3. Generate specific, data-driven insights based on what you actually find
4. If needed, call _regenerate_approach(new_prompt, discovered_data) for a new strategy
5. Avoid generic statements - be specific about what you discover in THIS data
6. If the request involves code exploration, locate the relevant files and read them before acting
7. Log each exploration step and summarize what was actually found

IMPORTANT: When analyzing images or data:
- Actually read and process the files, don't just describe what they might contain
- Extract real statistics, patterns, and insights from the actual data
- If analyzing an image, get its real dimensions, color statistics, etc.
- For CSV/JSON data, compute real statistics, find actual patterns
- Always base your insights on what you ACTUALLY find, not generic knowledge

Remember: NO markdown formatting, NO code blocks, just pure executable Python."""
        
        logger.info(f"Generating context-aware code for: {prompt[:100]}...")
        code = self.llm.generate_code(enhanced_prompt, adaptive_system_prompt)
        code = self._clean_code(code)
        return code

    @staticmethod
    def _clean_code(code: str) -> str:
        """Remove markdown formatting from generated code"""
        # Remove any leading explanatory text before code
        lines = code.split('\n')
        
        # Find where actual code starts (imports, function definitions, or code statements)
        code_start = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if (stripped.startswith('import ') or 
                stripped.startswith('from ') or
                stripped.startswith('def ') or
                stripped.startswith('class ') or
                stripped.startswith('#') or
                (stripped and not stripped[0].isupper())):  # Likely code, not prose
                code_start = i
                break
        
        # Join from where code actually starts
        if code_start > 0:
            code = '\n'.join(lines[code_start:])
        
        # Remove ```python and ``` markers
        if '```python' in code:
            code = code.replace('```python\n', '').replace('\n```python', '')
        if '```' in code:
            # Handle code blocks
            if code.strip().startswith('```'):
                code = code[code.index('```')+3:]
            if code.strip().endswith('```'):
                code = code[:code.rindex('```')]
        
        # Remove any faasr imports or sys.path modifications that may have slipped in
        filtered_lines = []
        for line in code.split("\n"):
            stripped = line.strip()
            if stripped.startswith("import faasr") or stripped.startswith("from faasr import"):
                continue
            if stripped.startswith("sys.path.insert(") or "/opt/faas/faasr" in stripped:
                continue
            filtered_lines.append(line)

        return "\n".join(filtered_lines).strip()

    @staticmethod
    def validate_code_safety(code: str) -> bool:
        """
        Perform basic safety checks on generated code
        (Relaxed for serverless sandboxed environment)

        Arguments:
            code: Python code to validate

        Returns:
            True if code passes safety checks, False otherwise
        """
        dangerous_patterns = [
            "eval(",
            "exec(",
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
