import os

from dotenv import load_dotenv


load_dotenv()


def get_gemini_api_key() -> str:
    

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
    
        )

    return api_key