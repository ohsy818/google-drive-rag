"""
Supabase Storage document loader and processor
"""
import os
import tempfile
from typing import List, Optional, Dict, Any
from supabase import create_client
from document_loader import DocumentProcessor

class SupabaseStorageLoader:
    """Handles loading and processing of documents from Supabase Storage"""
    
    def __init__(self, supabase_url: str, supabase_key: str):
        """
        Initialize the Supabase Storage loader
        
        Args:
            supabase_url: Supabase project URL
            supabase_key: Supabase service role key
        """
        self.supabase = create_client(supabase_url, supabase_key)
        self.document_loader = DocumentProcessor()
        
    def list_files(self, bucket_name: str, folder_path: Optional[str] = None) -> List[dict]:
        """
        List files in Supabase Storage bucket
        
        Args:
            bucket_name: Name of the storage bucket
            folder_path: Optional folder path within the bucket
            
        Returns:
            List of file metadata dictionaries
        """
        try:
            if folder_path:
                response = self.supabase.storage.from_(bucket_name).list(folder_path)
            else:
                response = self.supabase.storage.from_(bucket_name).list()
                
            return response
        except Exception as e:
            print(f"Error listing files: {e}")
            return []
            
    def download_and_process_file(self, bucket_name: str, file_path: str) -> List[str]:
        """
        Download a file from Supabase Storage and process it using DocumentLoader
        
        Args:
            bucket_name: Name of the storage bucket
            file_path: Path to the file in the bucket
            
        Returns:
            List of processed text chunks
        """
        try:
            # Download file to temporary location
            response = self.supabase.storage.from_(bucket_name).download(file_path)
            
            # Get file extension
            file_extension = os.path.splitext(file_path)[1]
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
                temp_file.write(response)
                temp_file_path = temp_file.name
                
            # Process the file
            chunks = self.document_loader.load_document(temp_file_path)
            
            # Clean up temporary file
            os.unlink(temp_file_path)
            
            return chunks
            
        except Exception as e:
            print(f"Error downloading and processing file {file_path}: {e}")
            return []
            
    def process_bucket(self, bucket_name: str, folder_path: Optional[str] = None) -> List[str]:
        """
        Process all files in a Supabase Storage bucket
        
        Args:
            bucket_name: Name of the storage bucket
            folder_path: Optional folder path within the bucket
            
        Returns:
            List of all processed text chunks from all files
        """
        files = self.list_files(bucket_name, folder_path)
        all_chunks = []
        
        for file in files:
            if file.get('name'):  # Skip folders
                file_path = os.path.join(folder_path, file['name']) if folder_path else file['name']
                chunks = self.download_and_process_file(bucket_name, file_path)
                all_chunks.extend(chunks)
                
        return all_chunks
        
    def upload_to_storage(self, file_path: str, bucket_name: str, storage_path: str) -> dict:
        """
        Upload a file to Supabase Storage
        
        Args:
            file_path: Path to the local file
            bucket_name: Name of the storage bucket
            storage_path: Path where the file should be stored in the bucket
            
        Returns:
            dict: File metadata
        """
        try:
            with open(file_path, 'rb') as f:
                file_data = f.read()
                
            response = self.supabase.storage.from_(bucket_name).upload(
                storage_path,
                file_data
            )
            
            return {
                'file_path': storage_path,
                'bucket_name': bucket_name,
                'url': self.supabase.storage.from_(bucket_name).get_public_url(storage_path)
            }
            
        except Exception as e:
            print(f"Error uploading file to Supabase Storage: {e}")
            raise 