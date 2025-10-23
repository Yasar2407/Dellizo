import os
from dotenv import load_dotenv

load_dotenv()  # Load from .env

MONGODB_URI = os.getenv("MONGODB_URI")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CLASSIFIER_MODEL = os.getenv("CLASSIFIER_MODEL", "nateraw/bert-base-uncased-emotion")
PORT = int(os.getenv("PORT", 8000))


if GEMINI_API_KEY is None:
    raise ValueError("GEMINI_API_KEY is not set in the environment variables!")