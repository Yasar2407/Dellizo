***

# Emotion Analysis API

A lightweight **FastAPI-based Emotion Analysis API** that identifies the dominant emotion in a given text, retrieves similar examples using embeddings, and generates empathetic human-like insights using **Google Gemini**. The system also stores all analyses in **MongoDB** for summary statistics and visualization.

***

## Features

- **/analyze** (POST): Detect dominant emotion, confidence, and retrieve nearest examples with an AI-generated insight.  
- **/summary** (GET): Return aggregate statistics like total analyses, emotion distribution, average confidence, and recent analyses.
- **Emotion Classification:** Hugging Face **nateraw/bert-base-uncased-emotion** pipeline.  
- **Vector Search:** **FAISS** index built from 50 synthetic emotion-labeled sentences using **sentence-transformers/all-MiniLM-L6-v2**.  
- **Insight Generation:** Uses **Google Gemini 2.0 Flash** for empathetic reflection summaries.  
- **Database:** MongoDB Atlas for persistent storage with indexes for emotion and timestamps.  

***

## Folder Structure

```
app/
 ├── main.py                 # FastAPI entry point
 ├── routes.py               # API routes for /analyze and /summary
 ├── classifier.py           # Classifier model loader and label picker
 ├── config.py               # Environment variable management
 ├── db.py                   # MongoDB connection and indexing
 ├── embeddings_index.py     # FAISS embedding building and query logic
 ├── insight_gemini.py       # Insight generator via Gemini API
 ├── models.py               # Pydantic request/response models
 ├── utils.py                # Label mapping and helper utilities
 └── data/
     ├── examples.jsonl      # Labeled emotion examples
     └── faiss.index         # Prebuilt FAISS index
```

***

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/emotion-analysis-api.git
cd emotion-analysis-api
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate     # Mac/Linux
# or on Windows:
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the root directory:

```bash
GEMINI_API_KEY="your_google_genai_api_key"
MONGODB_URI="your_mongodb_connection_string"
EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"
CLASSIFIER_MODEL="nateraw/bert-base-uncased-emotion"
PORT=8000
```

### 5. Build FAISS index

```bash
python app/build_index.py
```

### 6. Run the API server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:  
`http://localhost:8000/docs`

***

## Example Usage

### Analyze endpoint

```bash
curl -X POST "http://localhost:8000/analyze" \
     -H "Content-Type: application/json" \
     -d '{"text": "I feel so grateful for my family’s support."}'
```

**Example Response:**
```json
{
  "emotion": "love",
  "confidence": 0.91,
  "examples": [
    {"text": "I feel loved and accepted for who I am.", "label": "love", "score": 0.87},
    {"text": "My heart feels warm thinking of my family.", "label": "love", "score": 0.84},
    {"text": "I care deeply for the people around me.", "label": "love", "score": 0.81}
  ],
  "insight": "The message conveys warmth and connection, reflecting love and gratitude."
}
```

### Summary endpoint

```bash
curl -X GET "http://localhost:8000/summary"
```

**Example Response:**
```json
{
  "total_texts": 53,
  "emotion_distribution": {"joy": 28.3, "sadness": 17.0, "love": 22.6, "anger": 9.4, "neutral": 22.7},
  "avg_confidence": 0.88,
  "last_5_analyses": [
    {"text": "Feeling calm and balanced.", "predicted_emotion": "neutral", "confidence": 0.82, "timestamp": "2025-10-23T10:22:48Z"}
  ]
}
```

***

## Design and Modeling Notes

### Emotion Classification
- Model: **nateraw/bert-base-uncased-emotion** fine-tuned on emotion classification dataset.
- The output label is mapped to one of seven canonical categories:
  - joy, sadness, anger, fear, love, surprise, disgust, neutral
- Uses helper function `map_label_to_emotion()` to normalize variations (e.g., happiness → joy, anxiety → fear).

### Embedding Retrieval
- Embeddings generated via **sentence-transformers/all-MiniLM-L6-v2** (lightweight transformer suited for semantic similarity).
- FAISS is used for nearest-neighbor retrieval for contextual examples.

### Insight Generation
- Uses **Gemini 2.0 Flash** through `google-genai` to generate 1–2 empathetic sentences summarizing the text and emotion.

**Prompt Example:**
```
"You are an expert in emotional intelligence. Analyze this text and summarize it with empathy."
```

### Data & Storage
- **MongoDB Atlas** stores analyses with fields:
  `text`, `predicted_emotion`, `confidence`, `examples`, `insight`, `timestamp`
- Indexed by `predicted_emotion` and `timestamp` for fast summaries.

***

