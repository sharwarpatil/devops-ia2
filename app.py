from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
from fastapi.middleware.cors import CORSMiddleware
# Load pretrained sentiment-analysis pipeline
MODEL_ID = "distilbert-base-uncased-finetuned-sst-2-english"
app = FastAPI(title="Sentiment Analysis API", version="1.0")

sentiment_model = pipeline("sentiment-analysis", model=MODEL_ID)

class TextIn(BaseModel):
    text: str


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # in prod, lock this down
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID}

@app.post("/predict")
def predict(inp: TextIn):
    result = sentiment_model(inp.text)[0]
    return {
        "text": inp.text,
        "label": result["label"],        # POSITIVE / NEGATIVE
        "score": round(float(result["score"]), 4)
    }
