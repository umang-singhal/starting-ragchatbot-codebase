import warnings
import logging
warnings.filterwarnings("ignore", message="resource_tracker: There appear to be.*")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os

from config import config
from rag_system import RAGSystem

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.logging.level),
    format=config.logging.format,
    datefmt=config.logging.date_format
)

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Course Materials RAG System", root_path="")

# Add trusted host middleware for proxy
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]
)

# Enable CORS with proper settings for proxy
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Initialize RAG system
rag_system = RAGSystem(config)

# Pydantic models for request/response
class QueryRequest(BaseModel):
    """Request model for course queries"""
    query: str
    session_id: Optional[str] = None

class QueryResponse(BaseModel):
    """Response model for course queries"""
    answer: str
    sources: List[Dict[str, Any]]  # Each source has 'name' and optional 'link'
    session_id: str

class CourseStats(BaseModel):
    """Response model for course statistics"""
    total_courses: int
    course_titles: List[str]

class NewSessionResponse(BaseModel):
    """Response model for new session creation"""
    session_id: str

# API Endpoints

@app.post("/api/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Process a query and return response with sources"""
    logger.info("Received query: %s (session_id: %s)", request.query[:100], request.session_id)
    try:
        # Create session if not provided
        session_id = request.session_id
        if not session_id:
            session_id = rag_system.session_manager.create_session()
            logger.debug("Created new session: %s", session_id)

        # Process query using RAG system
        answer, sources = rag_system.query(request.query, session_id)

        logger.info("Query completed successfully (session_id: %s, sources: %d)", session_id, len(sources))
        return QueryResponse(
            answer=answer,
            sources=sources,
            session_id=session_id
        )
    except Exception as e:
        logger.error("Error processing query: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/courses", response_model=CourseStats)
async def get_course_stats():
    """Get course analytics and statistics"""
    logger.debug("Fetching course statistics")
    try:
        analytics = rag_system.get_course_analytics()
        logger.info("Returning course stats: %d courses", analytics["total_courses"])
        return CourseStats(
            total_courses=analytics["total_courses"],
            course_titles=analytics["course_titles"]
        )
    except Exception as e:
        logger.error("Error fetching course stats: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/session/new", response_model=NewSessionResponse)
async def create_new_session():
    """Create a new conversation session"""
    logger.debug("Creating new session via API")
    try:
        session_id = rag_system.session_manager.create_session()
        logger.info("Created new session: %s", session_id)
        return NewSessionResponse(session_id=session_id)
    except Exception as e:
        logger.error("Error creating new session: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup_event():
    """Load initial documents on startup"""
    # Test LLM connection first
    logger.info("Testing LLM connection...")
    success, message = rag_system.ai_generator.test_connection()
    if success:
        logger.info("LLM connection test: %s", message)
    else:
        logger.error("LLM connection test failed: %s", message)
        raise HTTPException(status_code=500, detail="LLM connection test failed")
    
    # Load documents
    docs_path = "../docs"
    logger.info("Loading initial documents from %s", docs_path)
    if os.path.exists(docs_path):
        try:
            courses, chunks = rag_system.add_course_folder(docs_path, clear_existing=False)
            logger.info("Successfully loaded %d courses with %d chunks", courses, chunks)
        except Exception as e:
            logger.error("Error loading documents: %s", e, exc_info=True)
    else:
        logger.warning("Documents path does not exist: %s", docs_path)

# Custom static file handler with no-cache headers for development
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path


class DevStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        if isinstance(response, FileResponse):
            # Add no-cache headers for development
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response
    
    
# Serve static files for the frontend
app.mount("/", StaticFiles(directory="../frontend", html=True), name="static")