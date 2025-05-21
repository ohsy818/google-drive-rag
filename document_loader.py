import os
import uuid
from typing import List, Optional
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader,
    UnstructuredExcelLoader,
    UnstructuredFileLoader,
    PyPDFLoader,
    TextLoader
)

class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

    def load_document(self, file_path: str) -> Optional[List[Document]]:
        """Load a document from a file."""
        try:
            file_extension = os.path.splitext(file_path)[1].lower()
            
            if file_extension == '.txt':
                loader = TextLoader(file_path)
            elif file_extension == '.docx':
                loader = UnstructuredWordDocumentLoader(file_path)
            elif file_extension == '.pptx':
                loader = UnstructuredPowerPointLoader(file_path)
            elif file_extension == '.xlsx':
                loader = UnstructuredExcelLoader(file_path)
            elif file_extension == '.pdf':
                loader = PyPDFLoader(file_path)
            else:
                print(f"Unsupported file type: {file_extension}")
                return None

            documents = loader.load()
            
            # Add basic metadata including UUID
            doc_id = str(uuid.uuid4())  # Generate a single UUID for the document
            for doc in documents:
                doc.metadata.update({
                    "id": doc_id,
                    "source": file_path,
                    "file_name": os.path.basename(file_path),
                    "file_type": file_extension[1:]  # Remove the dot
                })
            
            return documents
        except Exception as e:
            print(f"Error loading document {file_path}: {e}")
            return None

    def process_documents(self, documents: List[Document], metadata: dict = None) -> List[Document]:
        """Process documents by splitting them into chunks and adding metadata."""
        try:
            print(f"Processing {len(documents)} documents...")
            # Add additional metadata if provided
            if metadata:
                for doc in documents:
                    doc.metadata.update(metadata)
            
            # Split documents into chunks
            chunks = self.text_splitter.split_documents(documents)
            
            # Add chunk information to metadata
            for i, chunk in enumerate(chunks):
                chunk_id = str(uuid.uuid4())  # Generate a unique ID for each chunk
                chunk.metadata.update({
                    "chunk_id": chunk_id,
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                })
            
            return chunks
        except Exception as e:
            print(f"Error processing documents: {e}")
            return []

    def process_directory(self, directory_path: str, metadata: dict = None) -> List[Document]:
        """Process all documents in a directory."""
        all_documents = []
        
        for root, _, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                print(f"Processing file: {file_path}")
                documents = self.load_document(file_path)
                
                if documents:
                    print(f"Loaded {len(documents)} documents from {file_path}")
                    all_documents.extend(documents)
        
        if all_documents:
            print(f"Processing {len(all_documents)} documents...")
            chunks = self.process_documents(all_documents, metadata)
            print(f"Created {len(chunks)} chunks")
            return chunks
        
        return []

# Example usage
if __name__ == "__main__":
    processor = DocumentProcessor()
    documents = processor.process_directory("./documents")
    print(f"Processed {len(documents)} document chunks") 