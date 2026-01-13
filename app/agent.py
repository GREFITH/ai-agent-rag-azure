import json
import logging
from typing import Dict, List, Optional, Any
from openai import AzureOpenAI
from app.config import settings
from app.rag import RAGPipeline
from app.memory import SessionMemory

logger = logging.getLogger(__name__)


class AIAgent:
    """AI Agent with decision-making capabilities, tool calling, and memory."""
    
    def __init__(self):
        self.client = AzureOpenAI(
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            api_key=settings.AZURE_OPENAI_API_KEY,
            api_version=settings.AZURE_OPENAI_API_VERSION
        )
        self.rag_pipeline = None
        self.memory = SessionMemory()
        self.tools = self._define_tools()
        
    def _define_tools(self) -> List[Dict]:
        """Define tools available to the agent"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "search_documents",
                    "description": "Search through company documents to find relevant information. Use this when the user asks about policies, procedures, guidelines, or any information that would be in company documents.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query to find relevant documents"
                            }
                        },
                        "required": ["query"]
                    }
                }
            }
        ]
    
    async def initialize(self):
        """Initialize the RAG pipeline"""
        logger.info("Initializing RAG pipeline...")
        self.rag_pipeline = RAGPipeline(self.client)
        await self.rag_pipeline.initialize()
        logger.info("RAG pipeline initialized successfully")
    
    async def process_query(self, query: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Process a user query through the agent pipeline"""
        try:
            # Get conversation history
            history = self.memory.get_history(session_id) if session_id else []
            
            # Prepare messages for the agent
            messages = self._prepare_messages(query, history)
            
            # First agent call - decision making
            logger.info("Agent making decision on query processing...")
            
            # Build parameters for GPT-5 compatibility
            api_params = {
                "model": settings.AZURE_OPENAI_DEPLOYMENT_NAME,
                "messages": messages,
                "tools": self.tools,
                "tool_choice": "auto",
                "max_completion_tokens": settings.MAX_TOKENS
            }
            
            # Only add temperature if it's 1.0 (GPT-5 requirement)
            if settings.TEMPERATURE == 1.0:
                api_params["temperature"] = 1.0
            
            response = self.client.chat.completions.create(**api_params)
            
            response_message = response.choices[0].message
            tool_calls = response_message.tool_calls
            
            sources = []
            reasoning = "Direct answer from LLM"
            
            # If agent decides to use tools
            if tool_calls:
                logger.info(f"Agent calling {len(tool_calls)} tool(s)")
                messages.append(response_message)
                
                # Execute tool calls
                for tool_call in tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    
                    logger.info(f"Calling tool: {function_name} with args: {function_args}")
                    
                    if function_name == "search_documents":
                        # Search documents using RAG
                        search_results = await self.rag_pipeline.search(function_args["query"])
                        sources = search_results["sources"]
                        context = search_results["context"]
                        
                        reasoning = "Answer based on document search"
                        
                        # Add tool response to messages
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": function_name,
                            "content": json.dumps({
                                "context": context,
                                "sources": sources
                            })
                        })
                
                # Second agent call with tool results
                logger.info("Agent generating final response with tool results...")
                
                # Build parameters for second call
                final_params = {
                    "model": settings.AZURE_OPENAI_DEPLOYMENT_NAME,
                    "messages": messages,
                    "max_completion_tokens": settings.MAX_TOKENS
                }
                
                # Only add temperature if it's 1.0
                if settings.TEMPERATURE == 1.0:
                    final_params["temperature"] = 1.0
                
                final_response = self.client.chat.completions.create(**final_params)
                
                answer = final_response.choices[0].message.content
            else:
                # Direct answer without tools
                logger.info("Agent providing direct answer")
                answer = response_message.content
            
            # Update memory
            if session_id:
                self.memory.add_message(session_id, "user", query)
                self.memory.add_message(session_id, "assistant", answer)
            
            return {
                "answer": answer,
                "sources": sources,
                "reasoning": reasoning
            }
            
        except Exception as e:
            logger.error(f"Error in agent processing: {str(e)}")
            raise
    
    def _prepare_messages(self, query: str, history: List[Dict]) -> List[Dict]:
        """Prepare messages for the LLM including system prompt and history"""
        system_prompt = """You are an intelligent AI assistant that helps users find information.

When a user asks a question:
1. Determine if you can answer directly with your general knowledge, OR
2. If the question is about specific company policies, procedures, documents, or requires factual information from documents, use the search_documents tool.

For document-based questions, ALWAYS use the search_documents tool to ensure accuracy.

When you receive search results:
- Use the provided context to answer accurately
- Cite the sources when relevant
- If the context doesn't contain the answer, say so clearly
- Be concise but comprehensive

Maintain a helpful, professional tone."""

        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history
        messages.extend(history[-settings.MAX_CONVERSATION_HISTORY:])
        
        # Add current query
        messages.append({"role": "user", "content": query})
        
        return messages
    
    def clear_session(self, session_id: str):
        """Clear conversation history for a session"""
        self.memory.clear_session(session_id)