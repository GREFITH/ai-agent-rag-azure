from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import logging
from datetime import datetime

from app.agent import AIAgent
from app.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI Agent System",
    description="AI Agent with RAG capabilities for document-based Q&A",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize AI Agent (singleton)
ai_agent = None


class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    source: List[str]
    session_id: Optional[str] = None
    timestamp: str
    reasoning: Optional[str] = None


@app.on_event("startup")
async def startup_event():
    """Initialize the AI agent on startup"""
    global ai_agent
    try:
        logger.info("Initializing AI Agent...")
        ai_agent = AIAgent()
        await ai_agent.initialize()
        logger.info("AI Agent initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize AI Agent: {str(e)}")
        raise


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "AI Agent System API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat()
    }


@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    """Main endpoint to query the AI agent"""
    try:
        if not ai_agent:
            raise HTTPException(status_code=503, detail="AI Agent not initialized")
        
        logger.info(f"Processing query: {request.query[:100]}...")
        
        # Process the query through the agent
        result = await ai_agent.process_query(
            query=request.query,
            session_id=request.session_id
        )
        
        response = QueryResponse(
            answer=result["answer"],
            source=result.get("sources", []),
            session_id=request.session_id,
            timestamp=datetime.utcnow().isoformat(),
            reasoning=result.get("reasoning")
        )
        
        logger.info(f"Query processed successfully. Sources: {len(response.source)}")
        return response
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.post("/clear_session")
async def clear_session(session_id: str):
    """Clear conversation history for a session"""
    try:
        if not ai_agent:
            raise HTTPException(status_code=503, detail="AI Agent not initialized")
        
        ai_agent.clear_session(session_id)
        return {"message": f"Session {session_id} cleared successfully"}
        
    except Exception as e:
        logger.error(f"Error clearing session: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error clearing session: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)