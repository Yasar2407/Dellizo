import os
from google import genai
from .config import GEMINI_API_KEY

client = genai.Client(api_key=GEMINI_API_KEY)

def generate_insight(text: str, emotion: str, confidence: float, examples: list) -> str:
    prompt = (
        f"You are an expert in emotional intelligence. "
        f"Analyze this text and summarize it in 1-2 sentences with empathy.\n\n"
        f"Text: {text}\n"
        f"Predicted Emotion: {emotion} (confidence {confidence:.2f})\n"
        f"Similar Examples: {[e['text'] for e in examples]}\n\n"
        f"Return exactly 1-2 sentences describing the emotion and giving a short reflection."
    )

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )
    return getattr(response, "text", str(response))
