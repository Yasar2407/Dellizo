import asyncio
from datetime import datetime
from fastapi import APIRouter, HTTPException
from app.models import AnalyzeRequest, AnalyzeResponse, SummaryResponse
from app.config import EMBEDDING_MODEL, CLASSIFIER_MODEL
from app.db import analyses_coll, ensure_indexes
from app.classifier import load_classifier, pick_best_label
from app.embeddings_index import EmbeddingIndex
from app.insight_gemini import generate_insight
from app.utils import map_label_to_emotion

router = APIRouter()

# Global singletons
CLASSIFIER = None
INDEXER = None


@router.on_event("startup")
async def startup_event():
    """Load models and ensure DB indexes."""
    global CLASSIFIER, INDEXER
    if CLASSIFIER is None:
        CLASSIFIER = await asyncio.to_thread(load_classifier, CLASSIFIER_MODEL)
    if INDEXER is None:
        INDEXER = EmbeddingIndex(EMBEDDING_MODEL)
        INDEXER.build_from_jsonl("app/data/examples.jsonl", "app/data/faiss.index")

    # Run sync ensure_indexes safely in a thread
    await asyncio.to_thread(ensure_indexes)


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    """Analyze emotion of a text and generate insight."""
    text = req.text.strip()
    print(f"📝 Received text: {text}")

    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    try:
        # Run classification
        raw = await asyncio.to_thread(CLASSIFIER, text)
        print(f"🤖 Raw classifier output: {raw}")

        label, score = await asyncio.to_thread(pick_best_label, raw)
        print(f"🏷️ Picked label: {label}, Confidence score: {score}")

        emotion = map_label_to_emotion(label)
        confidence = float(score)
        print(f"💖 Mapped emotion: {emotion}, Confidence: {confidence:.2f}")
    except Exception as e:
        print("❌ Classification error:", e)
        raise HTTPException(status_code=500, detail="Failed to classify text")

    try:
        # Retrieve top-3 similar examples
        examples = await asyncio.to_thread(INDEXER.query, text, 3)
        print(f"🔍 Top-3 similar examples: {examples}")
    except Exception as e:
        print("⚠️ Example retrieval error:", e)
        examples = []

    # Generate insight using Gemini
    try:
        insight = await asyncio.to_thread(generate_insight, text, emotion, confidence, examples)
        print(f"💡 Generated insight: {insight}")
    except Exception as e:
        print("⚠️ Gemini Insight Error:", e)
        insight = f"The text mainly expresses {emotion} (confidence {confidence:.2f})."

    # Store in MongoDB safely
    try:
        doc = {
            "text": text,
            "predicted_emotion": emotion,
            "confidence": confidence,
            "examples": examples,
            "insight": insight,
            "timestamp": datetime.utcnow(),
        }
        await asyncio.to_thread(analyses_coll.insert_one, doc)
        print(f"💾 Document inserted into MongoDB: {doc}")
    except Exception as e:
        print("⚠️ MongoDB insert error:", e)

    return AnalyzeResponse(
        emotion=emotion,
        confidence=confidence,
        examples=examples,
        insight=insight,
    )


@router.get("/summary", response_model=SummaryResponse)
async def summary():
    """Get overall emotion summary from stored analyses."""
    try:
        total = await asyncio.to_thread(analyses_coll.count_documents, {})
    except Exception as e:
        print("⚠️ Count documents error:", e)
        total = 0

    # Aggregation for emotion distribution
    dist = {}
    try:
        pipeline = [{"$group": {"_id": "$predicted_emotion", "count": {"$sum": 1}}}]
        cursor = analyses_coll.aggregate(pipeline)
        for doc in cursor:
            dist[doc["_id"]] = doc["count"]
    except Exception as e:
        print("⚠️ Distribution aggregation error:", e)

    dist_pct = {k: v / total * 100 if total else 0 for k, v in dist.items()}

    # Average confidence
    avg_conf = 0.0
    try:
        cursor2 = analyses_coll.aggregate([{"$group": {"_id": None, "avg": {"$avg": "$confidence"}}}])
        for doc in cursor2:
            avg_conf = doc["avg"]
    except Exception as e:
        print("⚠️ Avg confidence aggregation error:", e)

    # Last 5 analyses
    last5 = []
    try:
        cursor3 = analyses_coll.find().sort("timestamp", -1).limit(5)
        for doc in cursor3:
            last5.append({
                "text": doc["text"],
                "predicted_emotion": doc["predicted_emotion"],
                "confidence": doc["confidence"],
                "timestamp": str(doc["timestamp"]),
            })
    except Exception as e:
        print("⚠️ Last 5 analyses fetch error:", e)

    return SummaryResponse(
        total_texts=total,
        emotion_distribution=dist_pct,
        avg_confidence=avg_conf,
        last_5_analyses=last5,
    )
