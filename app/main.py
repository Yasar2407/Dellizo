from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.API.emotionsAPI import router as emotions_router
import uvicorn

# Initialize FastAPI app
app = FastAPI(title="Emotion Analysis API (Gemini + MongoDB Atlas)")

# Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # Allow all origins (adjust for production)
    allow_credentials=True,
    allow_methods=["*"],        # Allow all HTTP methods
    allow_headers=["*"],        # Allow all headers
)

# ✅ Include Emotion Analysis Router
app.include_router(emotions_router, prefix="/api/v1", tags=["Emotion Analysis"])

# ✅ Main entry point
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
