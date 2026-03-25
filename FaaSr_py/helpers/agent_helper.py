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

    def generate_code(self, prompt: str, system_prompt: str, temperature: float = 0.2) -> str:
        """Generate Python code using OpenAI's API"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4.1",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
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

    def generate_code(self, prompt: str, system_prompt: str, temperature: float = 0.2) -> str:
        """Generate Python code using Claude's API"""
        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=3500,
                temperature=temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Claude API error: {e}")
            raise RuntimeError(f"Failed to generate code with Claude: {e}")


class AgentCodeGenerator:
    """Generates text responses from an LLM for agent orchestration."""

    def __init__(self, api_key: str, provider: str):
        self.provider_name = provider.lower()

        if self.provider_name == "openai":
            self.llm = OpenAIProvider(api_key)
        elif self.provider_name == "claude":
            self.llm = ClaudeProvider(api_key)
        else:
            raise ValueError(f"Unknown provider: {provider}. Use 'openai' or 'claude'")

    def generate_text(self, prompt: str, system_prompt: str, temperature: float = 0.2) -> str:
        logger.info("Generating text response with custom system prompt")
        logger.debug(f"generate_text system_prompt (first 300 chars): {(system_prompt or '')[:300]}")
        logger.debug(f"generate_text temperature: {temperature}")
        return self.llm.generate_code(prompt, system_prompt, temperature)


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
