from google import genai

from trustquery.config import get_gemini_api_key


def get_gemini_client() -> genai.Client:
    """Create and return an authenticated Gemini client."""

    return genai.Client(api_key=get_gemini_api_key())


def generate_text(prompt: str) -> str:
    """Generate a text response from Gemini."""

    client = get_gemini_client()

    interaction = client.interactions.create(
        model="gemini-3.6-flash",
        input=prompt,
    )

    return interaction.output_text