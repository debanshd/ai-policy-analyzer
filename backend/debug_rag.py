#!/usr/bin/env python3
"""
Debug script to test RAG functionality in isolation
"""

import asyncio
import time
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

async def test_minimal_rag():
    """Test minimal RAG setup"""
    print("🔍 Testing minimal RAG setup...")
    
    # Check if API key is loaded
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not found in environment")
        print("   Please check your .env file")
        return False
    else:
        print(f"✅ OpenAI API key loaded: {api_key[:8]}...{api_key[-8:]}")
    
    try:
        # 1. Create a simple document
        print("1. Creating test document...")
        content = """
        Policy Recommendations:
        1. Implement carbon pricing
        2. Invest in renewable energy
        3. Support green jobs
        4. Improve energy efficiency
        """
        
        doc = Document(
            page_content=content,
            metadata={'filename': 'test.txt', 'source': 'test'}
        )
        
        # 2. Create text splitter
        print("2. Setting up text splitter...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
        chunks = text_splitter.split_documents([doc])
        print(f"   Created {len(chunks)} chunks")
        
        # 3. Create embeddings (test with timeout)
        print("3. Testing embeddings...")
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", timeout=10, max_retries=1)
        
        # Test a single embedding
        start_time = time.time()
        test_embedding = await asyncio.get_event_loop().run_in_executor(
            None, 
            lambda: embeddings.embed_query("test query")
        )
        embed_time = time.time() - start_time
        print(f"   ✅ Embedding test completed in {embed_time:.2f}s")
        
        # 4. Create vectorstore
        print("4. Creating vectorstore...")
        start_time = time.time()
        vectorstore = Qdrant.from_documents(
            chunks,
            embeddings,
            location=":memory:",
            collection_name="test_collection"
        )
        vs_time = time.time() - start_time
        print(f"   ✅ Vectorstore created in {vs_time:.2f}s")
        
        # 5. Test retrieval
        print("5. Testing retrieval...")
        start_time = time.time()
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        results = retriever.get_relevant_documents("policy recommendations")
        retrieval_time = time.time() - start_time
        print(f"   ✅ Retrieved {len(results)} documents in {retrieval_time:.2f}s")
        
        # 6. Test LLM
        print("6. Testing LLM...")
        start_time = time.time()
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, timeout=15, max_retries=1)
        
        # Simple test query
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: llm.invoke("What are 3 policy recommendations? Be brief.")
        )
        llm_time = time.time() - start_time
        print(f"   ✅ LLM responded in {llm_time:.2f}s")
        print(f"   Response: {response.content[:100]}...")
        
        print("\n🎉 All components working!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_rag_chain():
    """Test a simple RAG chain"""
    print("\n🔗 Testing RAG chain...")
    
    try:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.runnables import RunnablePassthrough
        from langchain_core.output_parsers import StrOutputParser
        from operator import itemgetter
        
        # Quick setup
        content = "Policy recommendations: 1. Carbon pricing 2. Renewable energy 3. Green jobs"
        doc = Document(page_content=content, metadata={'source': 'test'})
        
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", timeout=10)
        vectorstore = Qdrant.from_documents([doc], embeddings, location=":memory:", collection_name="chain_test")
        retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
        
        # Create simple prompt
        prompt = ChatPromptTemplate.from_template("""
        Answer based on context: {context}
        Question: {question}
        Answer:
        """)
        
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, timeout=15)
        
        # Create chain
        chain = (
            {"context": itemgetter("question") | retriever, "question": itemgetter("question")}
            | RunnablePassthrough.assign(context=lambda x: "\n".join([doc.page_content for doc in x["context"]]))
            | {"response": prompt | llm | StrOutputParser(), "context": itemgetter("context")}
        )
        
        # Test the chain
        print("   Executing RAG chain...")
        start_time = time.time()
        
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: chain.invoke({"question": "What are the policy recommendations?"})
        )
        
        chain_time = time.time() - start_time
        print(f"   ✅ RAG chain completed in {chain_time:.2f}s")
        print(f"   Response: {result['response'][:150]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ RAG chain error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    print("🚀 RAG Debug Test")
    print("=" * 40)
    
    # Check environment setup
    print("🔧 Checking environment...")
    openai_key = os.getenv("OPENAI_API_KEY")
    tavily_key = os.getenv("TAVILY_API_KEY") 
    
    if openai_key:
        print(f"✅ OpenAI API key: {openai_key[:8]}...{openai_key[-8:]}")
    else:
        print("❌ OpenAI API key not found")
        return
        
    if tavily_key:
        print(f"✅ Tavily API key: {tavily_key[:8]}...{tavily_key[-8:]}")
    else:
        print("⚠️ Tavily API key not found (optional for RAG test)")
    
    print()
    
    # Test components individually
    components_ok = await test_minimal_rag()
    
    if components_ok:
        # Test full chain
        chain_ok = await test_rag_chain()
        
        if chain_ok:
            print("\n✅ All tests passed! RAG is working.")
            print("The issue might be in the agent orchestration.")
        else:
            print("\n❌ RAG chain failed.")
    else:
        print("\n❌ Basic components failed.")

if __name__ == "__main__":
    asyncio.run(main()) 