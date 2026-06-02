import os
from dotenv import load_dotenv

load_dotenv()

def get_llm_config(api_key: str) -> dict:
    """
    Returns the LLM config for AutoGen agents.
    Uses the user-provided Groq API key (BYOK pattern).
    """
    return {
        "config_list": [{
            "model": "llama-3.3-70b-versatile",
            "api_key": api_key,
            "base_url": "https://api.groq.com/openai/v1"
        }],
        "temperature": 0.5,
    }

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
