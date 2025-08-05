#!/usr/bin/env python3
"""
Task 5: Golden Dataset Evaluation - Minimal Execution
"""

import os
import sys
import random
import argparse
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd

load_dotenv()
sys.path.append(str(Path(__file__).parent.parent / "backend"))

from tools.rag_tool import RAGTool
from ragas.testset import TestsetGenerator
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader

def main():
    parser = argparse.ArgumentParser(description="Task 5: Golden Dataset Evaluation")
    parser.add_argument("--testset-size", type=int, default=10, help="Number of questions")
    args = parser.parse_args()
    
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY required")
    
    # Load documents for RAGAS generation only
    documents = []
    for file_path in Path("../documents").glob("*.pdf"):
        documents.extend(PyMuPDFLoader(str(file_path)).load())
    
    # Generate questions using RAGAS
    generator = TestsetGenerator(
        llm=LangchainLLMWrapper(ChatOpenAI(model="gpt-4o")),
        embedding_model=LangchainEmbeddingsWrapper(OpenAIEmbeddings())
    )
    
    testset = generator.generate_with_langchain_docs(
        documents=random.sample(documents, min(50, len(documents))),
        testset_size=args.testset_size
    )
    
    # Extract questions and display testset
    testset_df = testset.to_pandas()
    test_questions = list(testset_df['user_input'])
    print("RAGAS Testset:")
    print(testset_df.to_string())
    
    # Evaluate using backend RAG
    rag_tool = RAGTool()
    results = []
    
    for question in test_questions:
        result = rag_tool._run(question)
        answer = result.get("answer", "") if isinstance(result, dict) else str(result)
        contexts = result.get("relevant_chunks", []) if isinstance(result, dict) else []
        results.append({'question': question, 'answer': answer, 'contexts': len(contexts)})
    
    # Display results as DataFrame
    results_df = pd.DataFrame(results)
    print("\nEvaluation Results:")
    print(results_df.to_string())
    
    return 0

if __name__ == "__main__":
    exit(main()) 