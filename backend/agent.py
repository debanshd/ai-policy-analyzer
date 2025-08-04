import uuid
import time
import json
from typing import Dict, Any, List, Optional, AsyncGenerator, Iterator
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.vectorstores import Qdrant

from config import settings
from models import StreamingUpdate, ToolType, AgentResponse
from tools.rag_tool import RAGTool
from tools.tavily_tool import TavilyTool
from tools.data_commons_tool import DataCommonsTool
from utils.document_processor import DocumentProcessor

class MultiSourceAnalysisAgent:
    """
    Main agent that orchestrates RAG, Data Commons, and Tavily tools
    to provide comprehensive analysis with real-time streaming
    """
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.model_name, 
            temperature=settings.temperature,
            timeout=30,  # 30 second timeout
            max_retries=2
        )
        self.session_stores: Dict[str, Dict[str, Any]] = {}
        
        # Initialize tools (create them without API calls during import)
        self.rag_tool = None
        self.tavily_tool = None
        self.data_commons_tool = None
        
        # Initialize document processor
        self.document_processor = DocumentProcessor()
        
        # Agent prompt for orchestration
        self.agent_prompt = self._create_agent_prompt()
        
        # Initialize tools lazily
        self._initialize_tools()
    
    def _initialize_tools(self):
        """Initialize tools - will fail hard if data_gemma is not available"""
        self.rag_tool = RAGTool()
        self.tavily_tool = TavilyTool()
        self.data_commons_tool = DataCommonsTool()  # Hard failure if data_gemma not available

    def process_files(self, files: List[Dict[str, Any]], session_id: str):
        """
        Process uploaded files and return vectorstore and file info
        
        Args:
            files: List of file dictionaries with 'filename', 'content', 'content_type'
            session_id: Session ID for this upload session
            
        Returns:
            Tuple of (vectorstore, file_info)
        """
        # Process files using document processor
        vectorstore = self.document_processor.process_uploaded_files(files, session_id)
        
        # Create file info for the session
        file_info = []
        for file_data in files:
            file_info.append({
                'filename': file_data['filename'],
                'size': file_data['size'],
                'content_type': file_data.get('content_type', 'unknown'),
                'processed_at': time.time()
            })
        
        return vectorstore, file_info

    def _create_agent_prompt(self) -> ChatPromptTemplate:
        """Create the main agent prompt for orchestration"""
        AGENT_TEMPLATE = """You are the Multi-Source Analysis Agent, an intelligent research assistant for policy analysts. Your role is to provide comprehensive answers by analyzing user-uploaded documents, enriching findings with live statistical data, and providing broader web context.

Your workflow:
1. **Analyze User Context**: Use RAG to find relevant information in uploaded documents
2. **Augment with Live Data**: Extract key entities and fetch relevant statistical data
3. **Enrich with Web Context**: Search for definitions, news, and broader context

Current session context:
- User has uploaded: {uploaded_files}
- Available tools: RAG (document analysis), Data Commons (statistical data), Tavily (web search)

User question: {question}

Please provide a comprehensive analysis by combining insights from all available sources. Be specific about which sources you're drawing from and provide a well-structured response.
"""
        return ChatPromptTemplate.from_template(AGENT_TEMPLATE)

    def set_session_vectorstore(self, session_id: str, vectorstore: Qdrant, file_info: List[Dict]):
        """Store vectorstore and file info for a session"""
        if session_id not in self.session_stores:
            self.session_stores[session_id] = {
                'created_at': time.time(),
                'files': []
            }
        
        self.session_stores[session_id]['vectorstore'] = vectorstore
        self.session_stores[session_id]['files'].extend(file_info)
        self.session_stores[session_id]['last_updated'] = time.time()
        
        # Update RAG tool with new vectorstore for this session
        if session_id in self.session_stores and self.rag_tool:
            self.rag_tool.update_vectorstore(vectorstore)
    
    async def process_query_streaming(
        self, 
        question: str, 
        session_id: str
    ) -> AsyncGenerator[str, None]:
        """
        Process a user query with streaming updates
        
        Yields JSON-encoded StreamingUpdate objects
        """
        start_time = time.time()
        tools_used = []
        all_sources = []
        
        try:
            # Check if session has documents
            if session_id not in self.session_stores:
                yield self._create_update("error", "No documents uploaded for this session. Please upload documents first.")
                return
            
            session_info = self.session_stores[session_id]
            uploaded_files = [f.get('filename', 'Unknown') for f in session_info.get('files', [])]
            
            yield self._create_update("thought", "Starting comprehensive analysis...")
            
            # Step 1: RAG Analysis
            yield self._create_update("tool", "Analyzing uploaded documents...", ToolType.RAG_TOOL)
            
            if not self.rag_tool:
                yield self._create_update("error", "RAG tool not initialized")
                return
                
            print(f"🔍 Starting RAG analysis for: {question[:50]}...")  # Debug
            rag_result = self.rag_tool._run(question)
            print(f"✅ RAG analysis completed")  # Debug
            tools_used.append(ToolType.RAG_TOOL)
            
            if rag_result.get("error"):
                yield self._create_update("error", f"RAG analysis failed: {rag_result['error']}")
                return
            
            rag_answer = rag_result.get("answer", "")
            relevant_chunks = rag_result.get("relevant_chunks", [])
            
            # Add document sources
            for chunk in relevant_chunks:
                filename = chunk.get('metadata', {}).get('filename', 'Unknown document')
                if filename not in all_sources:
                    all_sources.append(filename)
            
            yield self._create_update("result", f"✓ Found relevant information in {len(relevant_chunks)} document sections")
            
            # Step 2: Data Commons Analysis with data_gemma
            yield self._create_update("tool", "Fetching real statistical data from Data Commons...", ToolType.DATA_COMMONS_TOOL)
                
            print(f"📊 Starting Data Commons analysis with data_gemma...")  # Debug
            # Create context-rich query combining original question with RAG insights
            enriched_query = f"{question}\n\nContext from documents: {rag_answer[:200]}..."
            print(f"📊 Query: {enriched_query[:100]}...")  # Debug
            
            # Hard failure if data_gemma fails - no simulation mode fallback
            data_commons_result = self.data_commons_tool._run(query=enriched_query)
            tools_used.append(ToolType.DATA_COMMONS_TOOL)
            print(f"✅ Data Commons analysis completed")  # Debug
            
            methodology = data_commons_result.get("methodology", "unknown")
            data_points_count = len(data_commons_result.get("data_points", []))
            yield self._create_update("result", f"✓ Real statistical data retrieved via {methodology} methodology ({data_points_count} data points)")
            
            # Add data commons sources
            source = data_commons_result.get("source", "Data Commons")
            if source not in all_sources:
                all_sources.append(source)
            
            # Step 3: Web search for broader context
            yield self._create_update("tool", "Searching web for broader context and definitions...", ToolType.TAVILY_TOOL)
            
            if not self.tavily_tool:
                yield self._create_update("error", "Tavily tool not initialized")
                return
            
            # Extract key terms from question and RAG results for web search
            question_terms = [word for word in question.split() if len(word) > 3 and word.lower() not in ['what', 'where', 'when', 'how', 'why', 'are', 'the', 'and', 'for', 'with', 'this']]
            search_entities = question_terms[:2] if question_terms else [question.split()[0]]  # Use first 2 meaningful terms
            print(f"🌐 Starting web search for entities: {search_entities}")  # Debug
            
            try:
                tavily_result = self.tavily_tool.search_with_entities(search_entities, context="policy analysis research")
                tools_used.append(ToolType.TAVILY_TOOL)
                print(f"✅ Web search completed")  # Debug
            except Exception as e:
                print(f"⚠️ Web search failed: {e}")
                tavily_result = {
                    "summary": f"Web search encountered an issue: {str(e)}",
                    "urls": [],
                    "search_results": []
                }
                yield self._create_update("result", "⚠️ Web search completed with warnings")
            
            # Add web sources
            web_urls = tavily_result.get("urls", [])
            all_sources.extend([url for url in web_urls[:5] if url not in all_sources])  # Limit URLs
            
            yield self._create_update("result", f"✓ Found {len(tavily_result.get('search_results', []))} web sources")
            
            # Step 4: Synthesize final answer
            yield self._create_update("thought", "Synthesizing comprehensive analysis...")
            
            print(f"🎯 Starting final synthesis...")  # Debug
            synthesis_prompt = self._create_synthesis_prompt()
            # Format Data Commons insights from new structure
            data_insights = ""
            data_points = data_commons_result.get("data_points", [])
            methodology = data_commons_result.get("methodology", "unknown")
            
            if data_points:
                data_insights = f"Statistical Data (via {methodology}):\n"
                for i, point in enumerate(data_points[:3], 1):  # Show top 3 points
                    value = point.get("value", "No data")
                    source = point.get("source", "Data Commons")
                    data_insights += f"{i}. {value} (Source: {source})\n"
            else:
                data_insights = data_commons_result.get("summary", "No statistical data available")
            
            synthesis_input = {
                "question": question,
                "rag_analysis": rag_answer,
                "data_insights": data_insights,
                "web_context": tavily_result.get("summary", ""),
                "uploaded_files": ", ".join(uploaded_files)
            }
            
            final_answer = self.llm.invoke(
                synthesis_prompt.format(**synthesis_input)
            ).content
            print(f"✅ Final synthesis completed")  # Debug
            
            processing_time = time.time() - start_time
            
            # Create final response
            agent_response = AgentResponse(
                answer=final_answer,
                sources=all_sources,
                tools_used=tools_used,
                session_id=session_id,
                processing_time=processing_time
            )
            
            yield self._create_update("result", "✓ Analysis complete!", metadata=agent_response.dict())
            
        except Exception as e:
            yield self._create_update("error", f"Agent processing error: {str(e)}")
    
    def _create_synthesis_prompt(self) -> ChatPromptTemplate:
        """Create prompt for final synthesis"""
        SYNTHESIS_TEMPLATE = """You are synthesizing a comprehensive analysis for a policy analyst. Combine the insights from multiple sources to provide a thorough, well-structured answer.

**User Question**: {question}

**Document Analysis** (from uploaded files: {uploaded_files}):
{rag_analysis}

**Statistical Data Insights**:
{data_insights}

**Web Context & Broader Information**:
{web_context}

**Instructions**:
1. Provide a comprehensive answer that integrates all three sources of information
2. Clearly indicate when you're drawing from uploaded documents vs. external sources
3. Structure your response logically with clear sections or points
4. Highlight any interesting connections or contradictions between sources
5. Include specific details and evidence to support your analysis
6. Maintain a professional, analytical tone suitable for policy research

**Comprehensive Analysis**:
"""
        return ChatPromptTemplate.from_template(SYNTHESIS_TEMPLATE)
    
    def _create_update(
        self, 
        update_type: str, 
        content: str, 
        tool: Optional[ToolType] = None, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a JSON-encoded streaming update"""
        update = StreamingUpdate(
            type=update_type,
            content=content,
            tool=tool,
            metadata=metadata
        )
        return json.dumps(update.dict()) + "\n"
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a session"""
        return self.session_stores.get(session_id)
    
    def cleanup_old_sessions(self, max_age_hours: int = 24):
        """Clean up old sessions to prevent memory leaks"""
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        sessions_to_remove = []
        for session_id, session_data in self.session_stores.items():
            if current_time - session_data.get('created_at', 0) > max_age_seconds:
                sessions_to_remove.append(session_id)
        
        for session_id in sessions_to_remove:
            del self.session_stores[session_id]
    
    def create_new_session(self) -> str:
        """Create a new session ID"""
        return str(uuid.uuid4())

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and its associated data"""
        if session_id in self.session_stores:
            del self.session_stores[session_id]
            return True
        return False

    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all active sessions with their metadata"""
        sessions = []
        for session_id, session_data in self.session_stores.items():
            session_info = {
                'session_id': session_id,
                'created_at': session_data.get('created_at', 0),
                'last_updated': session_data.get('last_updated', session_data.get('created_at', 0)),
                'files': len(session_data.get('files', [])),
                'file_list': [f['filename'] for f in session_data.get('files', [])]
            }
            sessions.append(session_info)
        return sessions 