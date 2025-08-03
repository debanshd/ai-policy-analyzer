import os
import tempfile
import uuid
from typing import List, Dict, Any, Optional
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant

from config import settings
from models import DocumentChunk

class DocumentProcessor:
    """Handles processing of uploaded documents (PDFs and images)"""
    
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(model=settings.embedding_model)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap
        )
        
    def process_uploaded_files(self, files: List[Dict[str, Any]], session_id: str) -> Qdrant:
        """
        Process uploaded files and create a Qdrant vectorstore
        
        Args:
            files: List of file dictionaries with 'filename', 'content', 'content_type'
            session_id: Session ID for this upload session
            
        Returns:
            Qdrant vectorstore containing the processed documents
        """
        all_documents = []
        
        for file_info in files:
            filename = file_info['filename']
            content = file_info['content']
            content_type = file_info['content_type']
            
            try:
                # Save file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix) as tmp_file:
                    tmp_file.write(content)
                    tmp_path = tmp_file.name
                
                # Process based on file type
                if filename.lower().endswith('.pdf'):
                    documents = self._process_pdf(tmp_path, filename)
                elif filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    documents = self._process_image(tmp_path, filename)
                elif filename.lower().endswith(('.txt', '.md')):
                    documents = self._process_text(tmp_path, filename)
                else:
                    raise ValueError(f"Unsupported file type: {filename}")
                
                # Add session metadata
                for doc in documents:
                    doc.metadata.update({
                        'session_id': session_id,
                        'source_filename': filename,
                        'file_id': str(uuid.uuid4())
                    })
                
                all_documents.extend(documents)
                
            finally:
                # Clean up temporary file
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
        
        if not all_documents:
            raise ValueError("No documents were successfully processed")
        
        # Split documents into chunks
        chunked_documents = self.text_splitter.split_documents(all_documents)
        
        # Create Qdrant vectorstore
        vectorstore = self._create_vectorstore(chunked_documents, session_id)
        
        return vectorstore
    
    def _process_pdf(self, file_path: str, filename: str) -> List[Document]:
        """Process a PDF file and extract text"""
        try:
            loader = PyMuPDFLoader(file_path)
            documents = loader.load()
            
            # Add filename to metadata
            for doc in documents:
                doc.metadata['filename'] = filename
                doc.metadata['file_type'] = 'pdf'
                
            return documents
        except Exception as e:
            raise ValueError(f"Failed to process PDF {filename}: {str(e)}")
    
    def _process_image(self, file_path: str, filename: str) -> List[Document]:
        """Process an image file using OCR (simplified for now)"""
        try:
            # For now, create a placeholder document with image metadata
            # In a full implementation, you would use OCR tools like pytesseract
            with Image.open(file_path) as img:
                width, height = img.size
                format = img.format
                
            content = f"Image file: {filename}\nDimensions: {width}x{height}\nFormat: {format}"
            
            doc = Document(
                page_content=content,
                metadata={
                    'filename': filename,
                    'file_type': 'image',
                    'width': width,
                    'height': height,
                    'format': format
                }
            )
            
            return [doc]
        except Exception as e:
            raise ValueError(f"Failed to process image {filename}: {str(e)}")
    
    def _process_text(self, file_path: str, filename: str) -> List[Document]:
        """Process a text file (.txt, .md)"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            doc = Document(
                page_content=content,
                metadata={
                    'filename': filename,
                    'file_type': 'text',
                    'encoding': 'utf-8',
                    'length': len(content)
                }
            )
            
            return [doc]
        except Exception as e:
            raise ValueError(f"Failed to process text file {filename}: {str(e)}")
    
    def _create_vectorstore(self, documents: List[Document], session_id: str) -> Qdrant:
        """Create a Qdrant vectorstore from documents"""
        try:
            collection_name = f"{settings.qdrant_collection_name}_{session_id}"
            
            vectorstore = Qdrant.from_documents(
                documents,
                self.embeddings,
                location=settings.qdrant_location,
                collection_name=collection_name
            )
            
            return vectorstore
        except Exception as e:
            raise ValueError(f"Failed to create vectorstore: {str(e)}")
    
    def documents_to_chunks(self, documents: List[Document]) -> List[DocumentChunk]:
        """Convert LangChain documents to DocumentChunk models"""
        chunks = []
        for doc in documents:
            chunk = DocumentChunk(
                content=doc.page_content,
                metadata=doc.metadata,
                page_number=doc.metadata.get('page', None)
            )
            chunks.append(chunk)
        return chunks

def validate_file_upload(filename: str, file_size: int) -> bool:
    """Validate uploaded file"""
    # Check file extension
    file_ext = Path(filename).suffix.lower()
    if file_ext not in settings.allowed_file_types:
        raise ValueError(f"File type {file_ext} not allowed. Allowed types: {settings.allowed_file_types}")
    
    # Check file size
    if file_size > settings.max_file_size:
        raise ValueError(f"File size {file_size} exceeds maximum allowed size {settings.max_file_size}")
    
    return True 