from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List
import uvicorn
from predictor import predictor

app = FastAPI(title="Weather Related Disease Prediction API")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionInput(BaseModel):
    data: Dict[str, Any]

@app.get("/")
async def root():
    return {"message": "Disease Prediction API is running"}

@app.post("/predict")
async def predict_disease(input_data: PredictionInput):
    try:
        result = predictor.predict(input_data.data)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/login")
async def login():
    # Mock login
    return {"status": "success", "message": "Logged in successfully"}

@app.post("/signup")
async def signup():
    # Mock signup
    return {"status": "success", "message": "Account created successfully"}

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
