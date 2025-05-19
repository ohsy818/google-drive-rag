"""
Google Drive document loader and processor
"""
import os
import io
import tempfile
from typing import List, Optional
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

from document_loader import DocumentLoader

SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

class GoogleDriveLoader:
    """Handles loading and processing of documents from Google Drive"""
    
    def __init__(self, credentials_path: str = 'credentials.json', token_path: str = 'token.json'):
        """
        Initialize the Google Drive loader
        
        Args:
            credentials_path: Path to the credentials.json file
            token_path: Path to save/load the token.json file
        """
        self.credentials_path = credentials_path
        self.token_path = token_path
        self.credentials = None
        self.service = None
        self.document_loader = DocumentLoader()
        
    def authenticate(self) -> None:
        """Authenticate with Google Drive API"""
        if os.path.exists(self.token_path):
            self.credentials = Credentials.from_authorized_user_file(self.token_path, SCOPES)
            
        if not self.credentials or not self.credentials.valid:
            if self.credentials and self.credentials.expired and self.credentials.refresh_token:
                self.credentials.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(self.credentials_path, SCOPES)
                self.credentials = flow.run_local_server(port=0)
                
            with open(self.token_path, 'w') as token:
                token.write(self.credentials.to_json())
                
        self.service = build('drive', 'v3', credentials=self.credentials)
        
    def list_files(self, folder_id: Optional[str] = None, file_types: Optional[List[str]] = None) -> List[dict]:
        """
        List files in Google Drive, optionally filtered by folder and file types
        
        Args:
            folder_id: Optional Google Drive folder ID to list files from
            file_types: Optional list of file MIME types to filter by
            
        Returns:
            List of file metadata dictionaries
        """
        if not self.service:
            self.authenticate()
            
        query_parts = []
        
        if folder_id:
            query_parts.append(f"'{folder_id}' in parents")
            
        if file_types:
            mime_types = [f"mimeType='{mime_type}'" for mime_type in file_types]
            query_parts.append(f"({' or '.join(mime_types)})")
            
        query_parts.append("trashed=false")
        query = " and ".join(query_parts)
        
        results = []
        page_token = None
        
        while True:
            try:
                response = self.service.files().list(
                    q=query,
                    spaces='drive',
                    fields='nextPageToken, files(id, name, mimeType)',
                    pageToken=page_token
                ).execute()
                
                results.extend(response.get('files', []))
                page_token = response.get('nextPageToken')
                
                if not page_token:
                    break
                    
            except Exception as e:
                print(f"Error listing files: {e}")
                break
                
        return results
        
    def download_and_process_file(self, file_id: str, file_name: str) -> List[str]:
        """
        Download a file from Google Drive and process it using DocumentLoader
        
        Args:
            file_id: Google Drive file ID
            file_name: Name of the file
            
        Returns:
            List of processed text chunks
        """
        if not self.service:
            self.authenticate()
            
        try:
            request = self.service.files().get_media(fileId=file_id)
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            
            while not done:
                status, done = downloader.next_chunk()
                
            fh.seek(0)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file_name)[1]) as temp_file:
                temp_file.write(fh.read())
                temp_file_path = temp_file.name
                
            chunks = self.document_loader.process_document(temp_file_path)
            
            os.unlink(temp_file_path)
            return chunks
            
        except Exception as e:
            print(f"Error downloading and processing file {file_name}: {e}")
            return []
            
    def process_folder(self, folder_id: str, file_types: Optional[List[str]] = None) -> List[str]:
        """
        Process all files in a Google Drive folder
        
        Args:
            folder_id: Google Drive folder ID
            file_types: Optional list of file MIME types to filter by
            
        Returns:
            List of all processed text chunks from all files
        """
        files = self.list_files(folder_id, file_types)
        all_chunks = []
        
        for file in files:
            chunks = self.download_and_process_file(file['id'], file['name'])
            all_chunks.extend(chunks)
            
        return all_chunks 