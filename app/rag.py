import os
import logging
from typing import List, Dict, Any
import faiss
import numpy as np
from pathlib import Path
import pickle
from pypdf import PdfReader

from app.config import settings

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Retrieval-Augmented Generation pipeline using FAISS"""
    
    def __init__(self, openai_client):
        self.client = openai_client
        self.index = None
        self.documents = []
        self.embeddings_cache = []
        
    async def initialize(self):
        """Initialize or load the vector store"""
        vectorstore_path = Path(settings.VECTOR_STORE_PATH)
        
        if vectorstore_path.exists() and (vectorstore_path / "index.faiss").exists():
            logger.info("Loading existing vector store...")
            self._load_vectorstore()
        else:
            logger.info("Creating new vector store...")
            await self._create_vectorstore()
    
    async def _create_vectorstore(self):
        """Create vector store from documents"""
        documents_path = Path(settings.DOCUMENTS_PATH)
        
        if not documents_path.exists():
            logger.warning(f"Documents path does not exist. Creating empty vector store.")
            documents_path.mkdir(parents=True, exist_ok=True)
            dimension = 1536
            self.index = faiss.IndexFlatL2(dimension)
            self._save_vectorstore()
            return
        
        # Load documents
        logger.info("Loading documents...")
        all_chunks = []
        
        for file_path in documents_path.glob("**/*"):
            if file_path.suffix.lower() == ".pdf":
                chunks = self._load_pdf(file_path)
                all_chunks.extend(chunks)
            elif file_path.suffix.lower() in [".txt", ".md"]:
                chunks = self._load_text(file_path)
                all_chunks.extend(chunks)
        
        if not all_chunks:
            logger.warning("No documents found.")
            dimension = 1536
            self.index = faiss.IndexFlatL2(dimension)
            self._save_vectorstore()
            return
        
        logger.info(f"Loaded {len(all_chunks)} chunks from documents")
        
        # Create embeddings
        logger.info("Creating embeddings...")
        embeddings = await self._create_embeddings([chunk["text"] for chunk in all_chunks])
        
        # Create FAISS index
        dimension = len(embeddings[0])
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings).astype('float32'))
        
        self.documents = all_chunks
        self.embeddings_cache = embeddings
        
        self._save_vectorstore()
        logger.info(f"Vector store created with {len(all_chunks)} chunks")
    
    def _load_pdf(self, file_path: Path) -> List[Dict]:
        """Load and chunk a PDF file"""
        try:
            reader = PdfReader(str(file_path))
            chunks = []
            
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text.strip():
                    page_chunks = self._chunk_text(text, file_path.name, page_num + 1)
                    chunks.extend(page_chunks)
            
            logger.info(f"Loaded {len(chunks)} chunks from {file_path.name}")
            return chunks
        except Exception as e:
            logger.error(f"Error loading PDF {file_path}: {str(e)}")
            return []
    
    def _load_text(self, file_path: Path) -> List[Dict]:
        """Load and chunk a text file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            chunks = self._chunk_text(text, file_path.name)
            logger.info(f"Loaded {len(chunks)} chunks from {file_path.name}")
            return chunks
        except Exception as e:
            logger.error(f"Error loading text file {file_path}: {str(e)}")
            return []
    
    def _chunk_text(self, text: str, source: str, page: int = None) -> List[Dict]:
        """Chunk text into smaller pieces"""
        chunks = []
        words = text.split()
        
        words_per_chunk = int(settings.CHUNK_SIZE * 0.75)
        overlap_words = int(settings.CHUNK_OVERLAP * 0.75)
        
        for i in range(0, len(words), words_per_chunk - overlap_words):
            chunk_words = words[i:i + words_per_chunk]
            chunk_text = " ".join(chunk_words)
            
            if chunk_text.strip():
                chunks.append({
                    "text": chunk_text,
                    "source": source,
                    "page": page,
                    "chunk_id": len(chunks)
                })
        
        return chunks
    
    async def _create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings for texts"""
        embeddings = []
        batch_size = 100
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                response = self.client.embeddings.create(
                    model=settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
                    input=batch
                )
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
                logger.info(f"Created embeddings for batch {i // batch_size + 1}")
            except Exception as e:
                logger.error(f"Error creating embeddings: {str(e)}")
                raise
        
        return embeddings
    
    async def search(self, query: str, top_k: int = None) -> Dict[str, Any]:
        """Search for relevant documents"""
        if top_k is None:
            top_k = settings.TOP_K_RESULTS
        
        if self.index is None or self.index.ntotal == 0:
            return {
                "context": "No documents available in the knowledge base.",
                "sources": []
            }
        
        # Create query embedding
        try:
            response = self.client.embeddings.create(
                model=settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
                input=[query]
            )
            query_embedding = np.array([response.data[0].embedding]).astype('float32')
        except Exception as e:
            logger.error(f"Error creating query embedding: {str(e)}")
            raise
        
        # Search in FAISS
        distances, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))
        
        # Retrieve documents
        results = []
        sources = set()
        
        for idx in indices[0]:
            if idx < len(self.documents):
                doc = self.documents[idx]
                results.append(doc["text"])
                source_info = doc["source"]
                if doc.get("page"):
                    source_info += f" (page {doc['page']})"
                sources.add(source_info)
        
        context = "\n\n".join(results)
        
        return {
            "context": context,
            "sources": list(sources)
        }
    
    def _save_vectorstore(self):
        """Save vector store to disk"""
        vectorstore_path = Path(settings.VECTOR_STORE_PATH)
        vectorstore_path.mkdir(parents=True, exist_ok=True)
        
        if self.index is not None:
            faiss.write_index(self.index, str(vectorstore_path / "index.faiss"))
        
        with open(vectorstore_path / "documents.pkl", "wb") as f:
            pickle.dump({
                "documents": self.documents,
                "embeddings": self.embeddings_cache
            }, f)
        
        logger.info(f"Vector store saved to {vectorstore_path}")
    
    def _load_vectorstore(self):
        """Load vector store from disk"""
        vectorstore_path = Path(settings.VECTOR_STORE_PATH)
        
        self.index = faiss.read_index(str(vectorstore_path / "index.faiss"))
        
        with open(vectorstore_path / "documents.pkl", "rb") as f:
            data = pickle.load(f)
            self.documents = data["documents"]
            self.embeddings_cache = data.get("embeddings", [])
        
        logger.info(f"Vector store loaded with {len(self.documents)} documents")