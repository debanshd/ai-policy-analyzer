import uuid
import time
import json
import asyncio
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
        """Initialize tools - data_gemma is optional"""
        self.rag_tool = RAGTool()
        self.tavily_tool = TavilyTool()
        try:
            self.data_commons_tool = DataCommonsTool()  # Will use fallback if data_gemma not available
        except Exception as e:
            print(f"⚠️ DataCommons tool initialization failed: {e}")
            self.data_commons_tool = None

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

    def _evaluate_answer_completeness(self, question: str, rag_answer: str, relevant_chunks: list) -> dict:
        """
        Evaluate if the RAG answer provides sufficient information to answer the user's question
        
        Returns:
            dict with 'is_complete' (bool), 'confidence' (float), and 'reasoning' (str)
        """
        evaluation_prompt = ChatPromptTemplate.from_template("""
You are an expert evaluator determining if a document-based answer fully addresses a user's question.

Question: {question}

Answer from Documents: {rag_answer}

Number of relevant document sections found: {chunk_count}

Your task: Determine if this answer is COMPLETE and SUFFICIENT to fully address the user's question.

Consider:
1. Does the answer directly address all aspects of the question?
2. Is the information specific and detailed enough?
3. Are there obvious gaps that would require external data sources?
4. Would the user be satisfied with this answer alone?

Respond with a JSON object containing:
{{
    "is_complete": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation of your assessment"
}}

Examples of questions that typically need external data:
- Current statistics, recent data, or real-time information
- Comparative analysis requiring broad datasets
- Questions about trends or changes over time
- Questions asking for specific numerical data not in documents

Examples of questions typically answerable from documents:
- Policy explanations, definitions, or interpretations
- Information explicitly contained in the uploaded documents
- Analysis of document content or summarization
- Questions about what the documents say or contain
""")
        
        try:
            evaluation_input = {
                "question": question,
                "rag_answer": rag_answer,
                "chunk_count": len(relevant_chunks)
            }
            
            response = self.llm.invoke(evaluation_prompt.format(**evaluation_input))
            
            # Parse JSON response
            import json
            try:
                result = json.loads(response.content)
                # Validate required fields
                if all(key in result for key in ['is_complete', 'confidence', 'reasoning']):
                    return result
            except json.JSONDecodeError:
                pass
                
            # Fallback if JSON parsing fails
            return {
                "is_complete": False,
                "confidence": 0.5,
                "reasoning": "Could not properly evaluate completeness"
            }
            
        except Exception as e:
            print(f"⚠️ Error evaluating answer completeness: {e}")
            return {
                "is_complete": False,
                "confidence": 0.5,
                "reasoning": f"Evaluation failed: {str(e)}"
            }

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
    
    def initialize_evaluation_vectorstore(self, session_id: str = "evaluation") -> str:
        """
        Initialize vectorstore with policy documents for evaluation
        
        Args:
            session_id: Session ID for evaluation (default: "evaluation")
            
        Returns:
            Session ID used
        """
        from pathlib import Path
        from langchain_core.documents import Document
        
        # Load policy documents from evals directory
        evals_path = Path(__file__).parent.parent / "evals" / "policy_documents"
        policy_files = [
            "policy_analysis_framework.txt",
            "economic_policy_statistics.txt"
        ]
        
        documents = []
        print(f"🔧 Loading policy documents for backend evaluation...")
        
        for policy_file in policy_files:
            policy_path = evals_path / policy_file
            if policy_path.exists():
                print(f"  Loading: {policy_file}")
                
                # Read the policy document
                with open(policy_path, 'r', encoding='utf-8') as f:
                    policy_content = f.read()
                
                # Create larger chunks by combining multiple paragraphs
                paragraphs = policy_content.split('\n\n')
                
                # Combine paragraphs into larger chunks of at least 500 characters
                current_chunk = ""
                chunk_count = 0
                
                for paragraph in paragraphs:
                    paragraph = paragraph.strip()
                    if paragraph:
                        current_chunk += paragraph + "\n\n"
                        
                        # If chunk is large enough, create a document
                        if len(current_chunk) > 500:
                            doc = Document(
                                page_content=current_chunk.strip(),
                                metadata={
                                    "source": policy_file,
                                    "chunk": chunk_count,
                                    "document_type": "policy_analysis"
                                }
                            )
                            documents.append(doc)
                            current_chunk = ""
                            chunk_count += 1
                
                # Add any remaining content as final chunk
                if current_chunk.strip():
                    doc = Document(
                        page_content=current_chunk.strip(),
                        metadata={
                            "source": policy_file,
                            "chunk": chunk_count,
                            "document_type": "policy_analysis"
                        }
                    )
                    documents.append(doc)
            else:
                print(f"  Warning: {policy_file} not found, skipping...")
        
        print(f"  Created {len(documents)} document chunks from policy analysis files")
        
        if not documents:
            raise ValueError("No policy documents found for evaluation")
        
        # Convert documents to file format expected by document processor
        file_data = []
        for i, doc in enumerate(documents):
            file_data.append({
                'filename': f"policy_chunk_{i}.txt",
                'content': doc.page_content.encode('utf-8'),
                'content_type': 'text/plain',
                'size': len(doc.page_content)
            })
        
        # Process files and create vectorstore using existing infrastructure
        vectorstore, file_info = self.process_files(file_data, session_id)
        self.set_session_vectorstore(session_id, vectorstore, file_info)
        
        print(f"  ✅ Backend initialized with {len(file_info)} policy document chunks")
        print(f"  📋 Session ID: {session_id}")
        
        return session_id
    
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
                await asyncio.sleep(0)  # Allow update to be sent immediately
                return
            
            session_info = self.session_stores[session_id]
            uploaded_files = [f.get('filename', 'Unknown') for f in session_info.get('files', [])]
            
            yield self._create_update("thought", "Starting comprehensive analysis...")
            await asyncio.sleep(0)  # Allow update to be sent immediately
            
            # Step 1: RAG Analysis
            yield self._create_update("tool", "Analyzing uploaded documents...", ToolType.RAG_TOOL)
            await asyncio.sleep(0)  # Allow update to be sent immediately
            
            if not self.rag_tool:
                yield self._create_update("error", "RAG tool not initialized")
                await asyncio.sleep(0)  # Allow update to be sent immediately
                return
                
            yield self._create_update("debug", f"🔍 Starting RAG analysis for: {question[:50]}...")
            await asyncio.sleep(0)  # Allow update to be sent immediately
            
            # Store updates from RAG tool and yield them as they come
            async def async_rag_callback(update_type: str, content: str, metadata=None):
                yield self._create_update(update_type, content, metadata=metadata)
                await asyncio.sleep(0)  # Allow update to be sent immediately
            
            # For now, collect updates synchronously but yield them individually
            rag_updates = []
            def rag_callback(update_type: str, content: str, metadata=None):
                rag_updates.append((update_type, content, metadata))
            
            rag_result = self.rag_tool._run(question, callback_handler=rag_callback)
            
            # Yield collected RAG updates individually with delays
            for update_type, content, metadata in rag_updates:
                yield self._create_update(update_type, content, metadata=metadata)
                await asyncio.sleep(0.1)  # Small delay to simulate real-time processing
            
            yield self._create_update("debug", "✅ RAG analysis completed")
            await asyncio.sleep(0)  # Allow update to be sent immediately
            tools_used.append(ToolType.RAG_TOOL)
            
            if rag_result.get("error"):
                yield self._create_update("error", f"RAG analysis failed: {rag_result['error']}")
                await asyncio.sleep(0)  # Allow update to be sent immediately
                return
            
            rag_answer = rag_result.get("answer", "")
            relevant_chunks = rag_result.get("relevant_chunks", [])
            
            # Add document sources
            for chunk in relevant_chunks:
                filename = chunk.get('metadata', {}).get('filename', 'Unknown document')
                if filename not in all_sources:
                    all_sources.append(filename)
            
            yield self._create_update("result", f"✓ Found relevant information in {len(relevant_chunks)} document sections")
            await asyncio.sleep(0)  # Allow update to be sent immediately
            
            # Evaluate if the RAG answer is complete
            yield self._create_update("thought", "Evaluating if document contains sufficient information...")
            await asyncio.sleep(0)  # Allow update to be sent immediately
            
            completeness_eval = self._evaluate_answer_completeness(question, rag_answer, relevant_chunks)
            is_complete = completeness_eval.get("is_complete", False)
            confidence = completeness_eval.get("confidence", 0.0)
            reasoning = completeness_eval.get("reasoning", "")
            
            yield self._create_update("debug", f"📋 Completeness evaluation: complete={is_complete}, confidence={confidence:.2f}", metadata={"confidence": confidence, "is_complete": is_complete})
            await asyncio.sleep(0)  # Allow update to be sent immediately
            yield self._create_update("debug", f"📋 Reasoning: {reasoning}", metadata={"reasoning": reasoning})
            await asyncio.sleep(0)  # Allow update to be sent immediately
            
            # Initialize default results for external tools
            data_commons_result = {"data_points": [], "source": "", "methodology": "", "summary": ""}
            tavily_result = {"summary": "", "urls": [], "search_results": []}
            
            # Determine if we should use external tools
            should_use_external_tools = (
                settings.require_external_tools or  # Force external tools if configured
                (settings.enable_external_tools and  # External tools enabled AND
                 (not is_complete or confidence < settings.completeness_threshold))  # (incomplete OR low confidence)
            )
            
            if should_use_external_tools:
                yield self._create_update("thought", f"Document analysis confidence: {confidence:.1%}. Enhancing with external sources...")
                await asyncio.sleep(0)  # Allow update to be sent immediately
                
                # Step 2: Data Commons Analysis with data_gemma
                yield self._create_update("tool", "Fetching real statistical data from Data Commons...", ToolType.DATA_COMMONS_TOOL)
                await asyncio.sleep(0)  # Allow update to be sent immediately
                    
                yield self._create_update("debug", "📊 Starting Data Commons analysis with data_gemma...")
                await asyncio.sleep(0)  # Allow update to be sent immediately
                # Create context-rich query combining original question with RAG insights
                enriched_query = f"{question}\n\nDocument context: {rag_answer[:300]}...\n\nPlease provide specific statistical data and trends with Data Commons URLs when available."
                yield self._create_update("debug", f"📊 Query: {enriched_query[:100]}...")
                await asyncio.sleep(0)  # Allow update to be sent immediately
                
                try:
                    # Store updates that will be collected from tool
                    tool_updates = []
                    
                    def tool_callback_handler(update_type: str, content: str, metadata=None):
                        """Collect tool updates"""
                        tool_updates.append((update_type, content, metadata))
                    
                    # Hard failure if data_gemma fails - no simulation mode fallback
                    data_commons_result = self.data_commons_tool._run(query=enriched_query, callback_handler=tool_callback_handler)
                    
                    # Yield collected updates
                    for update_type, content, metadata in tool_updates:
                        yield self._create_update(update_type, content, ToolType.DATA_COMMONS_TOOL, metadata)
                        await asyncio.sleep(0)  # Allow update to be sent immediately
                    
                    tools_used.append(ToolType.DATA_COMMONS_TOOL)
                    yield self._create_update("debug", "✅ Data Commons analysis completed")
                    await asyncio.sleep(0)  # Allow update to be sent immediately
                    
                    methodology = data_commons_result.get("methodology", "unknown")
                    data_points_count = len(data_commons_result.get("data_points", []))
                    yield self._create_update("result", f"✓ Real statistical data retrieved via {methodology} methodology ({data_points_count} data points)")
                    await asyncio.sleep(0)  # Allow update to be sent immediately
                    
                    # Add data commons sources
                    source = data_commons_result.get("source", "Data Commons")
                    if source not in all_sources:
                        all_sources.append(source)
                except Exception as e:
                    yield self._create_update("debug", f"⚠️ Data Commons failed: {e}")
                    await asyncio.sleep(0)  # Allow update to be sent immediately
                    yield self._create_update("result", "⚠️ Data Commons unavailable, continuing with document analysis")
                    await asyncio.sleep(0)  # Allow update to be sent immediately
                
                # Step 3: Web search for broader context
                yield self._create_update("tool", "Searching web for broader context and definitions...", ToolType.TAVILY_TOOL)
                await asyncio.sleep(0)  # Allow update to be sent immediately
                
                if self.tavily_tool:
                    # Extract key terms from question and RAG results for web search
                    question_terms = [word for word in question.split() if len(word) > 3 and word.lower() not in ['what', 'where', 'when', 'how', 'why', 'are', 'the', 'and', 'for', 'with', 'this']]
                    search_entities = question_terms[:2] if question_terms else [question.split()[0]]  # Use first 2 meaningful terms
                    yield self._create_update("debug", f"🌐 Starting web search for entities: {search_entities}")
                    await asyncio.sleep(0)  # Allow update to be sent immediately
                    
                    try:
                        # Store updates that will be collected from Tavily tool
                        tavily_updates = []
                        
                        def tavily_callback_handler(update_type: str, content: str, metadata=None):
                            """Collect Tavily tool updates"""
                            tavily_updates.append((update_type, content, metadata))
                        
                        tavily_result = self.tavily_tool.search_with_entities(search_entities, context="policy analysis research", callback_handler=tavily_callback_handler)
                        
                        # Yield collected updates
                        for update_type, content, metadata in tavily_updates:
                            yield self._create_update(update_type, content, ToolType.TAVILY_TOOL, metadata)
                            await asyncio.sleep(0)  # Allow update to be sent immediately
                        
                        tools_used.append(ToolType.TAVILY_TOOL)
                        yield self._create_update("debug", "✅ Web search completed")
                        await asyncio.sleep(0)  # Allow update to be sent immediately
                        
                        # Add web sources
                        web_urls = tavily_result.get("urls", [])
                        all_sources.extend([url for url in web_urls[:5] if url not in all_sources])  # Limit URLs
                        
                        yield self._create_update("result", f"✓ Found {len(tavily_result.get('search_results', []))} web sources")
                        await asyncio.sleep(0)  # Allow update to be sent immediately
                    except Exception as e:
                        yield self._create_update("debug", f"⚠️ Web search failed: {e}")
                        await asyncio.sleep(0)  # Allow update to be sent immediately
                        tavily_result = {
                            "summary": f"Web search encountered an issue: {str(e)}",
                            "urls": [],
                            "search_results": []
                        }
                        yield self._create_update("result", "⚠️ Web search completed with warnings")
                        await asyncio.sleep(0)  # Allow update to be sent immediately
                else:
                    yield self._create_update("result", "⚠️ Web search tool not available")
                    await asyncio.sleep(0)  # Allow update to be sent immediately
            else:
                yield self._create_update("thought", f"Document provides complete answer (confidence: {confidence:.1%}). Skipping external tools for faster response...")
                await asyncio.sleep(0)  # Allow update to be sent immediately
                yield self._create_update("result", f"✓ Complete answer found in documents - no external sources needed")
                await asyncio.sleep(0)  # Allow update to be sent immediately
            
            # Step 4: Synthesize final answer
            if should_use_external_tools:
                yield self._create_update("thought", "Synthesizing comprehensive analysis from all sources...")
                await asyncio.sleep(0)  # Allow update to be sent immediately
            else:
                yield self._create_update("thought", "Synthesizing final answer from document analysis...")
                await asyncio.sleep(0)  # Allow update to be sent immediately
            
            yield self._create_update("debug", "🎯 Starting final synthesis...")
            await asyncio.sleep(0)  # Allow update to be sent immediately
            synthesis_prompt = self._create_synthesis_prompt()
            
            # Format Data Commons insights from new structure
            data_insights = ""
            data_points = data_commons_result.get("data_points", [])
            methodology = data_commons_result.get("methodology", "")
            
            if data_points and should_use_external_tools:
                data_insights = f"Statistical Data (via {methodology}):\n"
                for i, point in enumerate(data_points[:3], 1):  # Show top 3 points
                    value = point.get("value", "No data")
                    source = point.get("source", "Data Commons")
                    url = point.get("url", "")
                    
                    if url:  # Only include URL if it exists and is not empty
                        data_insights += f"{i}. {value} (Source: {source} - URL: {url})\n"
                    else:
                        data_insights += f"{i}. {value} (Source: {source})\n"
            elif should_use_external_tools:
                data_insights = data_commons_result.get("summary", "No statistical data available")
            else:
                data_insights = "External data sources not consulted - answer based on document content only"
            
            # Handle web context
            web_context = ""
            if should_use_external_tools:
                web_summary = tavily_result.get("summary", "")
                web_urls = tavily_result.get("urls", [])
                
                if web_summary:
                    web_context = f"Web Search Results:\n{web_summary}\n"
                    if web_urls:
                        web_context += f"\nRelevant URLs: {', '.join(web_urls[:3])}"  # Show top 3 URLs
                else:
                    web_context = "No relevant web search results found"
            else:
                web_context = "Web search not performed - answer based on document content only"
            
            synthesis_input = {
                "question": question,
                "rag_analysis": rag_answer,
                "data_insights": data_insights,
                "web_context": web_context,
                "uploaded_files": ", ".join(uploaded_files)
            }
            
            final_answer = self.llm.invoke(
                synthesis_prompt.format(**synthesis_input)
            ).content
            yield self._create_update("debug", "✅ Final synthesis completed")
            await asyncio.sleep(0)  # Allow update to be sent immediately
            
            processing_time = time.time() - start_time
            
            # Create final response
            agent_response = AgentResponse(
                answer=final_answer,
                sources=all_sources,
                tools_used=tools_used,
                session_id=session_id,
                processing_time=processing_time
            )
            
            # Create final result message based on what was actually used
            if should_use_external_tools and len(tools_used) > 1:
                final_message = f"✓ Comprehensive analysis complete! Sources: {len(all_sources)} total ({', '.join(tool_type.value for tool_type in tools_used)})"
            else:
                final_message = f"✓ Document-based analysis complete! ({len(all_sources)} sources) - Faster response by focusing on uploaded content"
            
            yield self._create_update("result", final_message, metadata=agent_response.dict())
            await asyncio.sleep(0)  # Allow update to be sent immediately
            
        except Exception as e:
            yield self._create_update("error", f"Agent processing error: {str(e)}")
            await asyncio.sleep(0)  # Allow update to be sent immediately
    
    def _create_synthesis_prompt(self) -> ChatPromptTemplate:
        """Create prompt for final synthesis"""
        SYNTHESIS_TEMPLATE = """You are synthesizing a comprehensive analysis for a policy analyst. Combine insights from multiple sources to provide a thorough, well-structured answer with EXPLICIT source attribution.

**User Question**: {question}

## SOURCE 1: DOCUMENT ANALYSIS (PDF Files: {uploaded_files})
{rag_analysis}

## SOURCE 2: STATISTICAL DATA 
{data_insights}

## SOURCE 3: WEB RESEARCH
{web_context}

**CRITICAL INSTRUCTIONS FOR SOURCE ATTRIBUTION**:

1. **CLEARLY IDENTIFY SOURCE FOR EACH PIECE OF INFORMATION**:
   - Start each paragraph or claim by identifying the source
   - Use these formats: "According to the uploaded PDF documents..." OR "Data Commons statistical data shows..." OR "Web research indicates..."

2. **USE PRECISE INLINE CITATIONS**:
   - [PDF: filename] for document information
   - [Data Commons: URL] for statistical data ONLY when URLs are provided in SOURCE 2 above
   - [Data Commons] for statistical data when no URL is available
   - [Web: URL] for web search information ONLY when URLs are provided in SOURCE 3 above
   - [Web] for web search information when no URL is available

3. **STRUCTURE YOUR RESPONSE WITH SOURCE SECTIONS**:
   - **Document Insights**: What the PDF files reveal
   - **Statistical Evidence**: What Data Commons data shows  
   - **External Context**: What web research adds
   - **Synthesis**: How all sources combine to answer the question

4. **ONLY USE REAL URLs**: Only include URLs in citations if they are explicitly provided in the source sections above. Do not create or assume URLs.

5. **END WITH COMPLETE SOURCE LIST**: List all sources with clickable URLs

**Example Response Structure**:
## Document Insights
According to the uploaded PDF documents, Australia is a major lithium producer [PDF: IEA_report.pdf]...

## Statistical Evidence  
Data Commons statistical data shows production increasing 15% annually [Data Commons] (URL would only be included if provided in SOURCE 2 above)...

## External Context
Web research indicates that industry experts predict continued growth due to EV demand [Web] (URL would only be included if provided in SOURCE 3 above)...

## Analysis & Synthesis
Combining all sources...

**Your Comprehensive Analysis**:
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
        return json.dumps(update.dict())
    
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