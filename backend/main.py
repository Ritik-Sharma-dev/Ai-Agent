from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List
import uvicorn
import shutil
import os

# Import our AI logic
from retriever import rank_candidates
from ingest import ingest_resumes

app = FastAPI(title="TalentScout AI", version="2.0.0")

# Ensure data directory exists
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)

class JobRequest(BaseModel):
    job_description: str

@app.get("/")
def health_check():
    return {"status": "Active", "message": "TalentScout API is running!"}

@app.post("/api/rank-candidates")
def api_rank_candidates(request: JobRequest):
    if not request.job_description.strip():
        raise HTTPException(status_code=400, detail="Job description cannot be empty.")
    try:
        results = rank_candidates(request.job_description)
        if isinstance(results, dict) and "error" in results:
            raise HTTPException(status_code=500, detail=results["error"])
        return {"status": "success", "candidates": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- NEW ENDPOINT: Upload and Ingest Resumes ---
@app.post("/api/upload-resumes")
async def api_upload_resumes(files: List[UploadFile] = File(...)):
    try:
        saved_files = []
        # 1. Save the uploaded files to the data/ folder
        for file in files:
            file_path = os.path.join(DATA_DIR, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            saved_files.append(file.filename)
            
        # 2. Trigger the LangChain Ingestion process
        ingest_resumes() 
        
        return {
            "status": "success", 
            "message": f"Successfully processed {len(saved_files)} resumes into the database!",
            "files": saved_files
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("🚀 Starting FastAPI Server on http://localhost:8000")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)