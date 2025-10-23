from pydantic import BaseModel
from typing import List, Dict, Any

class AnalyzeRequest(BaseModel):
    text: str

class RetrievedExample(BaseModel):
    text: str
    label: str
    score: float

class AnalyzeResponse(BaseModel):
    emotion: str
    confidence: float
    examples: List[RetrievedExample]
    insight: str

class SummaryResponse(BaseModel):
    total_texts: int
    emotion_distribution: Dict[str, float]
    avg_confidence: float
    last_5_analyses: List[Dict[str, Any]]
