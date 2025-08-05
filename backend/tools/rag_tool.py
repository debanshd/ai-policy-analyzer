from typing import Dict, Any, List, Optional
from langchain_core.tools import BaseTool
from langchain_community.vectorstores import Qdrant
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from pydantic import Field

# Advanced retriever imports
from langchain.retrievers import ParentDocumentRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.storage import InMemoryStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate

from config import settings
from models import RAGResult, DocumentChunk, ToolType

class RAGTool(BaseTool):
    """
    Advanced Retrieval-Augmented Generation tool with multiple retrieval methods
    Supports: Naive, Sentence Window, Parent Document, and HyDE retrieval
    """
    
    name: str = "rag_tool"
    description: str = "Search and analyze uploaded documents using advanced RAG methods"
    
    # Declare fields
    vectorstore: Optional[Qdrant] = Field(default=None, exclude=True)
    llm: Optional[ChatOpenAI] = Field(default=None, exclude=True)
    rag_prompt: Optional[ChatPromptTemplate] = Field(default=None, exclude=True)
    retrieval_method: str = Field(default="naive", exclude=True)
    parent_retriever: Optional[ParentDocumentRetriever] = Field(default=None, exclude=True)
    compression_retriever: Optional[ContextualCompressionRetriever] = Field(default=None, exclude=True)
    
    def __init__(self, vectorstore: Optional[Qdrant] = None, retrieval_method: str = "naive", **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, 'vectorstore', vectorstore)
        object.__setattr__(self, 'retrieval_method', retrieval_method)
        object.__setattr__(self, 'llm', ChatOpenAI(
            model=settings.llm_model, 
            temperature=0,
            timeout=30,
            max_retries=2
        ))
        object.__setattr__(self, 'rag_prompt', self._create_rag_prompt())
        
        # Initialize advanced retrievers if needed
        if retrieval_method in ["parent_document", "sentence_window"]:
            self._setup_advanced_retrievers()
        
    def _setup_advanced_retrievers(self):
        """Setup advanced retrievers"""
        if not self.vectorstore:
            return
            
        try:
            # Parent Document Retriever
            if self.retrieval_method == "parent_document":
                parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
                child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
                store = InMemoryStore()
                
                parent_retriever = ParentDocumentRetriever(
                    vectorstore=self.vectorstore,
                    docstore=store,
                    child_splitter=child_splitter,
                    parent_splitter=parent_splitter,
                )
                object.__setattr__(self, 'parent_retriever', parent_retriever)
                
            # Sentence Window (Contextual Compression)
            elif self.retrieval_method == "sentence_window":
                base_retriever = self.vectorstore.as_retriever(search_kwargs={"k": 10})
                compressor = LLMChainExtractor.from_llm(
                    ChatOpenAI(model="gpt-4o-mini", temperature=0)
                )
                compression_retriever = ContextualCompressionRetriever(
                    base_compressor=compressor,
                    base_retriever=base_retriever
                )
                object.__setattr__(self, 'compression_retriever', compression_retriever)
                
        except Exception as e:
            print(f"Warning: Could not setup advanced retriever {self.retrieval_method}: {e}")
            object.__setattr__(self, 'retrieval_method', "naive")
    
    def _create_hypothetical_document(self, question: str) -> str:
        """Create hypothetical document for HyDE method"""
        hyde_prompt = PromptTemplate(
            input_variables=["question"],
            template="""Please write a passage to answer the question below. 
The passage should be informative and factual, as if it were extracted from a policy document.

Question: {question}

Passage:"""
        )
        
        hyde_chain = hyde_prompt | self.llm
        response = hyde_chain.invoke({"question": question})
        return response.content if hasattr(response, 'content') else str(response)
    
    def _get_retriever(self, question: str = None):
        """Get the appropriate retriever based on the method"""
        if not self.vectorstore:
            raise ValueError("No vectorstore available")
            
        if self.retrieval_method == "naive":
            return self.vectorstore.as_retriever(search_kwargs={"k": 5})
            
        elif self.retrieval_method == "parent_document":
            if self.parent_retriever:
                return self.parent_retriever
            else:
                print("Warning: Parent retriever not available, falling back to naive")
                return self.vectorstore.as_retriever(search_kwargs={"k": 5})
                
        elif self.retrieval_method == "sentence_window":
            if self.compression_retriever:
                return self.compression_retriever
            else:
                print("Warning: Compression retriever not available, falling back to naive")
                return self.vectorstore.as_retriever(search_kwargs={"k": 5})
                
        elif self.retrieval_method == "hyde":
            # For HyDE, we create a hypothetical document and search with it
            if question:
                hypothetical_doc = self._create_hypothetical_document(question)
                # Use the hypothetical document for better semantic matching
                hyde_retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
                return hyde_retriever
            else:
                return self.vectorstore.as_retriever(search_kwargs={"k": 5})
                
        else:
            print(f"Warning: Unknown retrieval method {self.retrieval_method}, using naive")
            return self.vectorstore.as_retriever(search_kwargs={"k": 5})
    
    def _create_rag_prompt(self) -> ChatPromptTemplate:
        """Create the RAG prompt template based on notebook patterns"""
        RAG_TEMPLATE = """You are a helpful and knowledgeable research assistant. Use the context provided below to answer the question accurately and comprehensively.

If you cannot find relevant information in the context, say that the information is not available in the uploaded documents.

Question:
{question}

Context from uploaded documents:
{context}

Answer:"""

        return ChatPromptTemplate.from_template(RAG_TEMPLATE)
    
    def _run(self, question: str, callback_handler=None, **kwargs) -> Dict[str, Any]:
        """
        Execute RAG with the configured retrieval method
        
        Args:
            question: The user's question
            callback_handler: Optional callback for streaming updates
            
        Returns:
            Dictionary containing answer and relevant chunks
        """
        import time
        
        def emit_update(update_type: str, content: str, metadata=None):
            """Helper to emit updates if callback handler is available"""
            if callback_handler:
                callback_handler(update_type, content, metadata)
            else:
                print(f"[{update_type.upper()}] {content}")
        
        if not self.vectorstore:
            emit_update("debug", "❌ No vectorstore available")
            return {
                "answer": "I don't have access to any documents to answer your question.",
                "relevant_chunks": [],
                "retrieval_method": self.retrieval_method
            }
            
        emit_update("debug", f"📚 Using {self.retrieval_method} retrieval method...")
        start_time = time.time()
        
        try:
            # Get the appropriate retriever
            retriever = self._get_retriever(question)
            retriever_time = time.time() - start_time
            emit_update("debug", f"   ✅ {self.retrieval_method.title()} retriever ready in {retriever_time:.2f}s")
            
            emit_update("debug", "🔗 Building RAG chain...")
            
            # Special handling for HyDE
            if self.retrieval_method == "hyde":
                # Create hypothetical document for better retrieval
                hypothetical_doc = self._create_hypothetical_document(question)
                emit_update("debug", f"🔮 Generated hypothetical document: {hypothetical_doc[:100]}...")
                
                # Build HyDE RAG chain
                rag_chain = (
                    {"context": lambda x: retriever.get_relevant_documents(hypothetical_doc), "question": itemgetter("question")}
                    | RunnablePassthrough.assign(context=itemgetter("context"))
                    | RunnablePassthrough.assign(
                        raw_docs=itemgetter("context"),
                        context=lambda x: self._format_context(x["context"])
                    )
                    | {"response": self.rag_prompt | self.llm, "context": itemgetter("context"), "raw_docs": itemgetter("raw_docs")}
                )
            else:
                # Standard RAG chain for other methods
                rag_chain = (
                    {"context": itemgetter("question") | retriever, "question": itemgetter("question")}
                    | RunnablePassthrough.assign(context=itemgetter("context"))
                    | RunnablePassthrough.assign(
                        raw_docs=itemgetter("context"),
                        context=lambda x: self._format_context(x["context"])
                    )
                    | {"response": self.rag_prompt | self.llm, "context": itemgetter("context"), "raw_docs": itemgetter("raw_docs")}
                )
            
            emit_update("debug", "⚙️ Executing RAG chain...")
            chain_start_time = time.time()
            
            # Execute the chain
            result = rag_chain.invoke({"question": question})
            
            chain_time = time.time() - chain_start_time
            emit_update("debug", f"   ✅ RAG chain executed in {chain_time:.2f}s")
            
            # Extract response content
            response_content = result["response"]
            if hasattr(response_content, 'content'):
                response_content = response_content.content
            
            # Extract relevant chunks from raw documents
            relevant_chunks = []
            if "raw_docs" in result:
                for doc in result["raw_docs"]:
                    relevant_chunks.append({
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "source": doc.metadata.get("source", "Unknown")
                    })
            
            total_time = time.time() - start_time
            emit_update("debug", f"🎯 Total RAG execution time: {total_time:.2f}s")
            
            return {
                "answer": response_content,
                "relevant_chunks": relevant_chunks,
                "retrieval_method": self.retrieval_method,
                "execution_time": total_time
            }
            
        except Exception as e:
            emit_update("error", f"RAG execution failed: {str(e)}")
            return {
                "answer": f"I encountered an error while processing your question: {str(e)}",
                "relevant_chunks": [],
                "retrieval_method": self.retrieval_method,
                "error": str(e)
            }
    
    def _format_context(self, documents: List) -> str:
        """Format retrieved documents into context string"""
        if not documents:
            return "No relevant context found."
        
        context_parts = []
        for i, doc in enumerate(documents[:5], 1):  # Limit to top 5 for token efficiency
            content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
            source = doc.metadata.get('source', 'Unknown') if hasattr(doc, 'metadata') else 'Unknown'
            context_parts.append(f"[Document {i} - {source}]\n{content}")
        
        return "\n\n".join(context_parts)
    
    def update_vectorstore(self, new_vectorstore: Qdrant):
        """Update the vectorstore for this tool"""
        object.__setattr__(self, 'vectorstore', new_vectorstore)
        
        # Re-setup advanced retrievers if needed
        if self.retrieval_method in ["parent_document", "sentence_window"]:
            self._setup_advanced_retrievers()
    
    def set_retrieval_method(self, method: str):
        """Change the retrieval method"""
        valid_methods = ["naive", "parent_document", "sentence_window", "hyde"]
        if method not in valid_methods:
            raise ValueError(f"Invalid retrieval method. Must be one of: {valid_methods}")
        
        object.__setattr__(self, 'retrieval_method', method)
        
        # Setup advanced retrievers if needed
        if method in ["parent_document", "sentence_window"]:
            self._setup_advanced_retrievers() 