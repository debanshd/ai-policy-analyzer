from typing import Dict, Any, List, Optional
from langchain_core.tools import BaseTool
from langchain_community.vectorstores import Qdrant
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from pydantic import Field

from config import settings
from models import RAGResult, DocumentChunk, ToolType

class RAGTool(BaseTool):
    """
    Retrieval-Augmented Generation tool for querying uploaded documents
    Based on the patterns from the Advanced Retrieval notebook
    """
    
    name: str = "rag_tool"
    description: str = "Search and analyze uploaded documents to answer questions using RAG"
    
    # Declare vectorstore as a Pydantic field
    vectorstore: Optional[Qdrant] = Field(default=None, exclude=True)
    llm: Optional[ChatOpenAI] = Field(default=None, exclude=True)
    rag_prompt: Optional[ChatPromptTemplate] = Field(default=None, exclude=True)
    
    def __init__(self, vectorstore: Optional[Qdrant] = None, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, 'vectorstore', vectorstore)
        object.__setattr__(self, 'llm', ChatOpenAI(
            model=settings.llm_model, 
            temperature=0,
            timeout=30,  # 30 second timeout
            max_retries=2
        ))
        object.__setattr__(self, 'rag_prompt', self._create_rag_prompt())
        
    def _create_rag_prompt(self) -> ChatPromptTemplate:
        """Create the RAG prompt template based on notebook patterns"""
        RAG_TEMPLATE = """You are a helpful and knowledgeable research assistant. Use the context provided below to answer the question accurately and comprehensively.

If you cannot find relevant information in the context, say that the information is not available in the uploaded documents.

Question:
{question}

Context from uploaded documents:
{context}

Provide a detailed answer based on the context, and mention which documents or sections you're referencing when possible.
"""
        return ChatPromptTemplate.from_template(RAG_TEMPLATE)
    
    def _run(self, question: str, **kwargs) -> Dict[str, Any]:
        """Execute the RAG tool"""
        import time
        print(f"🔍 RAG._run called with question: {question[:50]}...")
        
        try:
            if not self.vectorstore:
                print("❌ No vectorstore available")
                return {
                    "error": "No documents have been uploaded for this session",
                    "answer": "Please upload documents before asking questions.",
                    "relevant_chunks": [],
                    "confidence_score": 0.0
                }
            
            print("📚 Creating retriever...")
            start_time = time.time()
            
            # Create retriever with higher k for comprehensive results
            retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": 5}  # Reduced from 10 for speed
            )
            retriever_time = time.time() - start_time
            print(f"   ✅ Retriever created in {retriever_time:.2f}s")
            
            print("🔗 Building RAG chain...")
            # Build RAG chain using LCEL pattern from notebook
            rag_chain = (
                # Get context and question
                {"context": itemgetter("question") | retriever, "question": itemgetter("question")}
                # Format the context for the prompt
                | RunnablePassthrough.assign(
                    context=lambda x: self._format_context(x["context"])
                )
                # Generate response
                | {"response": self.rag_prompt | self.llm | StrOutputParser(), 
                   "context": itemgetter("context"),
                   "retrieved_docs": itemgetter("context")}
            )
            
            print("⚙️ Executing RAG chain...")
            chain_start_time = time.time()
            
            # Execute the chain
            result = rag_chain.invoke({"question": question})
            
            chain_time = time.time() - chain_start_time
            print(f"   ✅ RAG chain executed in {chain_time:.2f}s")
            
            # Convert retrieved documents to DocumentChunk format
            relevant_chunks = []
            if isinstance(result.get("retrieved_docs"), list):
                for doc in result["retrieved_docs"]:
                    if hasattr(doc, 'page_content') and hasattr(doc, 'metadata'):
                        chunk = DocumentChunk(
                            content=doc.page_content,
                            metadata=doc.metadata,
                            page_number=doc.metadata.get('page')
                        )
                        relevant_chunks.append(chunk)
            
            return {
                "answer": result["response"],
                "relevant_chunks": [chunk.dict() for chunk in relevant_chunks],
                "confidence_score": self._calculate_confidence_score(result["response"], relevant_chunks),
                "tool_used": ToolType.RAG_TOOL
            }
            
        except Exception as e:
            return {
                "error": f"RAG tool execution failed: {str(e)}",
                "answer": "Sorry, I encountered an error while searching the documents.",
                "relevant_chunks": [],
                "confidence_score": 0.0
            }
    
    def _format_context(self, documents: List[Any]) -> str:
        """Format retrieved documents into context string"""
        if not documents:
            return "No relevant documents found."
        
        formatted_context = []
        for i, doc in enumerate(documents, 1):
            if hasattr(doc, 'page_content') and hasattr(doc, 'metadata'):
                filename = doc.metadata.get('filename', 'Unknown')
                page = doc.metadata.get('page', 'N/A')
                content = doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content
                
                formatted_context.append(
                    f"[Document {i}] {filename} (Page {page}):\n{content}\n"
                )
        
        return "\n".join(formatted_context)
    
    def _calculate_confidence_score(self, answer: str, chunks: List[DocumentChunk]) -> float:
        """Calculate a simple confidence score based on retrieved chunks"""
        if not chunks:
            return 0.0
        
        # Simple heuristic: more chunks with content = higher confidence
        # In a production system, you'd use more sophisticated methods
        base_score = min(len(chunks) / 5.0, 1.0)  # Normalize to 0-1
        
        # Boost score if answer is substantial
        if len(answer) > 100:
            base_score *= 1.2
        
        return min(base_score, 1.0)
    
    async def _arun(self, question: str, **kwargs) -> Dict[str, Any]:
        """Async version of the tool"""
        return self._run(question, **kwargs)
    
    def update_vectorstore(self, new_vectorstore: Qdrant):
        """Update the vectorstore for this tool"""
        object.__setattr__(self, 'vectorstore', new_vectorstore) 