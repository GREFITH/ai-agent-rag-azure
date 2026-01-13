"""
Script to initialize the vector store from documents.
Run this after placing documents in the ./documents folder.
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import AzureOpenAI
from app.config import settings
from app.rag import RAGPipeline
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    """Initialize the vector store"""
    try:
        logger.info("Starting vector store initialization...")
        
        # Initialize Azure OpenAI client
        client = AzureOpenAI(
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            api_key=settings.AZURE_OPENAI_API_KEY,
            api_version=settings.AZURE_OPENAI_API_VERSION
        )
        
        # Create RAG pipeline
        rag = RAGPipeline(client)
        
        # Initialize (this will create the vector store)
        await rag.initialize()
        
        logger.info("✓ Vector store initialized successfully!")
        logger.info(f"  - Documents processed: {len(rag.documents)}")
        logger.info(f"  - Vector store location: {settings.VECTOR_STORE_PATH}")
        
    except Exception as e:
        logger.error(f"✗ Error initializing vector store: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())