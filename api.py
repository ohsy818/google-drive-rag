from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import os

from document_loader import DocumentProcessor
from google_drive_loader import GoogleDriveLoader
from supabase_storage_loader import SupabaseStorageLoader
from rag_chain import RAGChain

app = FastAPI(title="Document RAG API", description="API for document processing and querying")

class ProcessDocumentsRequest(BaseModel):
    input_dir: str
    storage_type: Optional[str] = "Local"

class ProcessGoogleDriveRequest(BaseModel):
    folder_id: str
    credentials_path: str
    storage_type: Optional[str] = "GoogleDrive"

class QueryRequest(BaseModel):
    question: str
    storage_type: Optional[str] = None

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, str]]

class SaveToDriveRequest(BaseModel):
    file_path: str
    folder_id: str
    credentials_path: str

class ProcessSupabaseStorageRequest(BaseModel):
    bucket_name: str
    folder_path: Optional[str] = None
    storage_type: Optional[str] = "SupabaseStorage"

class SaveToStorageRequest(BaseModel):
    file_path: str
    bucket_name: str
    storage_path: str

@app.post("/process")
async def process_documents(request: ProcessDocumentsRequest):
    """Process documents from the input directory"""
    if not os.path.exists(request.input_dir):
        raise HTTPException(status_code=404, detail=f"Directory {request.input_dir} does not exist")
    
    try:
        processor = DocumentProcessor()
        documents = processor.process_directory(request.input_dir, {"storage_type": request.storage_type})
        
        if not documents:
            return {"message": "No documents found to process"}
        
        rag = RAGChain()
        success = rag.process_and_store_documents(documents)
        
        if success:
            return {"message": "Successfully processed and stored documents"}
        else:
            raise HTTPException(status_code=500, detail="Failed to process and store documents")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process-drive")
async def process_google_drive(request: ProcessGoogleDriveRequest):
    """Process documents from Google Drive folder"""
    try:
        drive_loader = GoogleDriveLoader(credentials_path=request.credentials_path)
        supported_mime_types = [
            'application/vnd.google-apps.document',
            'application/vnd.google-apps.spreadsheet',
            'application/vnd.google-apps.presentation',
            'application/pdf',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            'application/vnd.openxmlformats-officedocument.presentationml.presentation',
            'text/plain'
        ]
        
        chunks = drive_loader.process_folder(request.folder_id, file_types=supported_mime_types)
        
        if not chunks:
            return {"message": "No documents found to process"}
        
        rag = RAGChain()
        documents = [{"page_content": chunk, "metadata": {"storage_type": request.storage_type}} for chunk in chunks]
        success = rag.process_and_store_documents(documents)
        
        if success:
            return {"message": "Successfully processed and stored documents from Google Drive"}
        else:
            raise HTTPException(status_code=500, detail="Failed to process and store documents from Google Drive")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
async def answer_question(request: QueryRequest):
    """Answer a question using the RAG system"""
    try:
        rag = RAGChain()
        metadata_filter = {"storage_type": request.storage_type} if request.storage_type else None
        result = rag.answer_question(request.question, metadata_filter)
        
        sources = []
        for doc in result["source_documents"]:
            sources.append({
                "source": doc.metadata.get("source", "Unknown source"),
                "content": doc.page_content[:200] + "..."
            })
        
        return QueryResponse(
            answer=result["answer"],
            sources=sources
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/save-to-drive")
async def save_to_drive(request: SaveToDriveRequest):
    """Save a file to Google Drive"""
    try:
        drive_loader = GoogleDriveLoader(credentials_path=request.credentials_path)
        result = drive_loader.save_to_google_drive(request.file_path, request.folder_id)
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process-storage")
async def process_supabase_storage(request: ProcessSupabaseStorageRequest):
    """Process documents from Supabase Storage bucket"""
    try:
        supabase_url = os.getenv('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_KEY')
        
        if not supabase_url or not supabase_key:
            raise HTTPException(status_code=500, detail="SUPABASE_URL and SUPABASE_KEY must be set in environment variables")
            
        storage_loader = SupabaseStorageLoader(supabase_url, supabase_key)
        chunks = storage_loader.process_bucket(request.bucket_name, request.folder_path)
        
        if not chunks:
            return {"message": "No documents found to process"}
        
        rag = RAGChain()
        documents = [{"page_content": chunk, "metadata": {"storage_type": request.storage_type}} for chunk in chunks]
        success = rag.process_and_store_documents(documents)
        
        if success:
            return {"message": "Successfully processed and stored documents from Supabase Storage"}
        else:
            raise HTTPException(status_code=500, detail="Failed to process and store documents from Supabase Storage")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/save-to-storage")
async def save_to_storage(request: SaveToStorageRequest):
    """Save a file to Supabase Storage"""
    try:
        supabase_url = os.getenv('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_KEY')
        
        if not supabase_url or not supabase_key:
            raise HTTPException(status_code=500, detail="SUPABASE_URL and SUPABASE_KEY must be set in environment variables")
            
        storage_loader = SupabaseStorageLoader(supabase_url, supabase_key)
        result = storage_loader.upload_to_storage(
            request.file_path,
            request.bucket_name,
            request.storage_path
        )
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)


"""
- 로컬 문서 처리
http POST http://localhost:8001/process input_dir="./documents"

- 구글 드라이브 문서 처리
http POST http://localhost:8001/process-drive folder_id="your_folder_id" credentials_path="./credentials.json"

- 질의
http POST http://localhost:8001/query question="프로젝트 A의 예산이 얼마 나왔지?"

- 특정 저장소 타입 지정 질의
http POST http://localhost:8001/query question="your_question" storage_type="Local"

- 구글 드라이브 파일 저장
http POST http://localhost:8001/save-to-drive file_path="./documents/your_file.pdf" folder_id="your_folder_id" credentials_path="./credentials.json"
"""
