"""Load provider configuration without exposing credentials."""
import os
from dotenv import load_dotenv

load_dotenv()


def get_gemini_api_key() -> str:
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        raise RuntimeError('GEMINI_API_KEY is not set in the environment')
    return api_key
