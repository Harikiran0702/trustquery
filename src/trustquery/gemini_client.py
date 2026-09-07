import time

from google import genai

from trustquery.config import get_gemini_api_key


def get_gemini_client() -> genai.Client:
    """Create and return an authenticated Gemini client."""
    return genai.Client(api_key=get_gemini_api_key())


def generate_text(
    prompt: str,
    max_retries: int = 3,
    initial_wait: float = 15.0,
) -> str:
    """Generate text with retry handling for transient Gemini errors."""
    client = get_gemini_client()

    for attempt in range(max_retries):
        try:
            interaction = client.interactions.create(
                model="gemini-3.6-flash",
                input=prompt,
            )

            return interaction.output_text

        except Exception as exc:
            message = str(exc).lower()

            quota_exhausted = (
                "quota exceeded" in message
                and "free_tier_requests" in message
            )

            if quota_exhausted:
                raise

            retryable = (
                "429" in message
                or "too_many_requests" in message
                or "500" in message
                or "high demand" in message
                or "internalservererror" in message
            )

            if not retryable:
                raise

            if attempt == max_retries - 1:
                raise

            wait_seconds = initial_wait * (2 ** attempt)

            print(
                f"Gemini temporary error. "
                f"Retrying in {wait_seconds:.0f} seconds..."
            )

            time.sleep(wait_seconds)

    raise RuntimeError("Gemini generation failed unexpectedly.")