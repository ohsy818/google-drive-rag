import os
import argparse
from pathlib import Path
from typing import Optional, Dict, Any

from document_loader import DocumentProcessor
from google_drive_loader import GoogleDriveLoader
from supabase_storage_loader import SupabaseStorageLoader
from rag_chain import RAGChain

def process_documents(input_dir: str, metadata: Optional[Dict[str, Any]] = None) -> None:
    """Process documents from the input directory and store them in the vector store"""
    print(f"Processing documents from {input_dir}...")
    
    if not os.path.exists(input_dir):
        print(f"Error: Directory {input_dir} does not exist")
        return
    
    processor = DocumentProcessor()
    documents = processor.process_directory(input_dir, metadata)
    print(f"Found {len(documents)} document chunks")
    
    if documents:
        rag = RAGChain()
        success = rag.process_and_store_documents(documents)
        if success:
            print("Successfully processed and stored documents")
        else:
            print("Failed to process and store documents")

def process_google_drive(folder_id: str, credentials_path: str, metadata: Optional[Dict[str, Any]] = None) -> None:
    """Process documents from Google Drive folder"""
    print(f"Processing documents from Google Drive folder {folder_id}...")
    
    try:
        drive_loader = GoogleDriveLoader(credentials_path=credentials_path)
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
        
        chunks = drive_loader.process_folder(folder_id, file_types=supported_mime_types)
        print(f"Found {len(chunks)} document chunks")
        
        if chunks:
            rag = RAGChain()
            documents = [{"page_content": chunk, "metadata": metadata or {}} for chunk in chunks]
            success = rag.process_and_store_documents(documents)
            if success:
                print("Successfully processed and stored documents from Google Drive")
            else:
                print("Failed to process and store documents from Google Drive")
                
    except Exception as e:
        print(f"Error processing Google Drive documents: {e}")

def process_supabase_storage(bucket_name: str, folder_path: Optional[str], metadata: Optional[Dict[str, Any]] = None) -> None:
    """Process documents from Supabase Storage bucket"""
    print(f"Processing documents from Supabase Storage bucket {bucket_name}...")
    
    try:
        supabase_url = os.getenv('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_KEY')
        
        if not supabase_url or not supabase_key:
            print("Error: SUPABASE_URL and SUPABASE_KEY must be set in environment variables")
            return
            
        storage_loader = SupabaseStorageLoader(supabase_url, supabase_key)
        chunks = storage_loader.process_bucket(bucket_name, folder_path)
        print(f"Found {len(chunks)} document chunks")
        
        if chunks:
            rag = RAGChain()
            documents = [{"page_content": chunk, "metadata": metadata or {}} for chunk in chunks]
            success = rag.process_and_store_documents(documents)
            if success:
                print("Successfully processed and stored documents from Supabase Storage")
            else:
                print("Failed to process and store documents from Supabase Storage")
                
    except Exception as e:
        print(f"Error processing Supabase Storage documents: {e}")

def answer_question(question: str, metadata_filter: Optional[Dict[str, Any]] = None) -> None:
    """Answer a question using the RAG system"""
    try:
        rag = RAGChain()
        result = rag.answer_question(question, metadata_filter)
        
        print("\nAnswer:", result["answer"])
        print("\nSources:")
        for doc in result["source_documents"]:
            print(f"\nSource: {doc.metadata.get('source', 'Unknown source')}")
            print(f"Content: {doc.page_content[:200]}...")
    except Exception as e:
        print(f"Error answering question: {e}")

def main():
    parser = argparse.ArgumentParser(description="Document-based RAG System")
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Process local documents command
    process_parser = subparsers.add_parser("process", help="Process local documents")
    process_parser.add_argument("--input_dir", required=True, help="Input directory containing documents")
    process_parser.add_argument("--storage_type", default="Local", help="Storage type (e.g., Local, GoogleDrive)")
    
    # Process Google Drive command
    drive_parser = subparsers.add_parser("process-drive", help="Process Google Drive documents")
    drive_parser.add_argument("--folder_id", required=True, help="Google Drive folder ID")
    drive_parser.add_argument("--credentials", required=True, help="Path to Google Drive credentials.json")
    drive_parser.add_argument("--storage_type", default="GoogleDrive", help="Storage type tag")
    
    # Process Supabase Storage command
    storage_parser = subparsers.add_parser("process-storage", help="Process Supabase Storage documents")
    storage_parser.add_argument("--bucket", required=True, help="Supabase Storage bucket name")
    storage_parser.add_argument("--folder", help="Optional folder path within the bucket")
    storage_parser.add_argument("--storage_type", default="SupabaseStorage", help="Storage type tag")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query the system")
    query_parser.add_argument("--question", required=True, help="Question to ask")
    query_parser.add_argument("--storage_type", help="Filter by storage type")
    
    args = parser.parse_args()
    
    if args.command == "process":
        metadata = {"storage_type": args.storage_type} if args.storage_type else None
        process_documents(args.input_dir, metadata)
    elif args.command == "process-drive":
        metadata = {"storage_type": args.storage_type} if args.storage_type else None
        process_google_drive(args.folder_id, args.credentials, metadata)
    elif args.command == "process-storage":
        metadata = {"storage_type": args.storage_type} if args.storage_type else None
        process_supabase_storage(args.bucket, args.folder, metadata)
    elif args.command == "query":
        metadata_filter = {"storage_type": args.storage_type} if args.storage_type else None
        answer_question(args.question, metadata_filter)
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 