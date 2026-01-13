from typing import Dict, List, Optional
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class SessionMemory:
    """Simple in-memory session management for conversation history."""
    
    def __init__(self):
        self.sessions: Dict[str, List[Dict]] = defaultdict(list)
        self.max_history = 20
    
    def add_message(self, session_id: str, role: str, content: str):
        """Add a message to session history"""
        if not session_id:
            return
        
        message = {"role": role, "content": content}
        self.sessions[session_id].append(message)
        
        # Trim history if too long
        if len(self.sessions[session_id]) > self.max_history:
            self.sessions[session_id] = self.sessions[session_id][-self.max_history:]
        
        logger.debug(f"Added {role} message to session {session_id}")
    
    def get_history(self, session_id: Optional[str]) -> List[Dict]:
        """Get conversation history for a session"""
        if not session_id:
            return []
        return self.sessions.get(session_id, [])
    
    def clear_session(self, session_id: str):
        """Clear all messages for a session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"Cleared session {session_id}")
    
    def get_session_count(self) -> int:
        """Get number of active sessions"""
        return len(self.sessions)