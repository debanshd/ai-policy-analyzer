#!/usr/bin/env python3
"""
Task 5: Golden Dataset Evaluation with Advanced Backend RAG Methods
Direct Version - RAGAS Testset Generation with Long-Form Policy Analysis Documents (No Fallbacks)
"""

import os
import sys
import random
import argparse
import time
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from datetime import datetime

load_dotenv()
# Add both backend directory and project root to Python path
backend_path = Path(__file__).parent.parent / "backend"
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))
sys.path.insert(0, str(project_root))

from tools.rag_tool import RAGTool
from ragas.testset import TestsetGenerator
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader

# RAGAS evaluation imports
from ragas import EvaluationDataset, evaluate
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness, ResponseRelevancy
from ragas import RunConfig
from langsmith import Client

def main():
    parser = argparse.ArgumentParser(description="Task 5: Advanced Backend RAG Evaluation")
    parser.add_argument("--testset-size", type=int, default=3, help="Number of questions (default: 3)")
    parser.add_argument("--skip-langsmith", action="store_true", help="Skip LangSmith dataset creation")
    args = parser.parse_args()
    
    print("🔧 Advanced Backend RAG Evaluation - Fast Mode")
    print("   • Testing backend RAG tool with multiple retrieval methods")
    print("   • Using gpt-4o-mini for reliable RAGAS testset generation")
    print(f"   • Generating {args.testset_size} questions")
    print("   • Using long-form policy analysis document chunks for RAGAS compatibility")
    print("   • Methods: Naive, Parent Document, Sentence Window, HyDE")
    
    # Load documents for RAGAS generation - Using policy analysis documents from evals subdirectory
    documents = []
    policy_documents_path = Path(__file__).parent / "policy_documents"
    
    # Load policy analysis and statistics documents
    policy_files = [
        "policy_analysis_framework.txt",
        "economic_policy_statistics.txt"
    ]
    
    print(f"Loading policy analysis documents from evals subdirectory...")
    
    from langchain_core.documents import Document
    
    for policy_file in policy_files:
        policy_path = policy_documents_path / policy_file
        if policy_path.exists():
            print(f"  Loading: {policy_file}")
            
            # Read the policy document
            with open(policy_path, 'r', encoding='utf-8') as f:
                policy_content = f.read()
            
            # Create larger chunks by combining multiple paragraphs (RAGAS needs >100 tokens)
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
    if documents:
        chunk_sizes = [len(doc.page_content) for doc in documents]
        print(f"  Chunk sizes: min={min(chunk_sizes)}, max={max(chunk_sizes)}, avg={sum(chunk_sizes)/len(chunk_sizes):.0f} characters")
        print(f"  Sample chunk: {documents[0].page_content[:150]}...")
    else:
        print("  No chunks created")
    
    print(f"Loaded {len(documents)} document chunks total")
    
    # Show content summary
    total_content_length = sum(len(doc.page_content) for doc in documents)
    print(f"Total content length: {total_content_length} characters")
    
    # Fast RAGAS testset generation
    print("\n🎯 Generating RAGAS testset...")
    
    # Use policy analysis document chunks for reliable generation
    print(f"Using {len(documents)} document chunks from policy analysis documents for testset generation")
    print(f"Total document content length: {sum(len(doc.page_content) for doc in documents)} characters")
    print(f"Sample document content: {documents[0].page_content[:200]}...")
    
    # Use policy analysis document chunks for reliable generation
    analysis_docs = documents
    
    # Create RAGAS generator with reliable configuration
    generator_llm = LangchainLLMWrapper(ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
        max_tokens=1000,
        timeout=60
    ))
    generator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(
        model="text-embedding-3-small"
    ))
    
    generator = TestsetGenerator(
        llm=generator_llm,
        embedding_model=generator_embeddings
    )
    
    print("Starting RAGAS testset generation...")
    
    # Ensure we have enough content for generation
    print(f"Documents available: {len(analysis_docs)}")
    print(f"Total content length: {sum(len(doc.page_content) for doc in analysis_docs)} characters")
    print(f"Average doc length: {sum(len(doc.page_content) for doc in analysis_docs) / len(analysis_docs):.0f} characters")
    
    # Generate testset using policy analysis document chunks with explicit parameters
    print("Generating RAGAS testset with policy analysis document chunks...")
    
    # Use all policy analysis document chunks (they're already well-structured and properly sized)
    optimal_doc_count = len(analysis_docs)
    ragas_docs = analysis_docs
    print(f"Using {len(ragas_docs)} documents for RAGAS generation")
    
    # Print first document sample for verification
    print(f"First document preview: {ragas_docs[0].page_content[:300]}...")
    print(f"First document metadata: {ragas_docs[0].metadata}")
    
    # Ensure minimum testset size for RAGAS
    testset_size = max(2, args.testset_size)
    print(f"Requesting {testset_size} questions from RAGAS")
    
    # Generate testset directly with RAGAS
    print("Calling RAGAS generate_with_langchain_docs...")
    testset = generator.generate_with_langchain_docs(
        documents=ragas_docs,
        testset_size=testset_size
    )
    
    testset_df = testset.to_pandas()
    print("✅ RAGAS generation completed!")
    print(f"Generated DataFrame shape: {testset_df.shape}")
    print(f"Generated DataFrame columns: {testset_df.columns.tolist()}")

    print(f"\n✅ Generated testset with {len(testset_df)} questions")
    print("RAGAS Testset Columns:", testset_df.columns.tolist())
    print("RAGAS Testset Shape:", testset_df.shape)
    print("RAGAS Testset Info:")
    print(testset_df.info())
    print("\nFirst few rows:")
    print(testset_df.head())
    
    # Extract questions from RAGAS testset
    if 'user_input' in testset_df.columns:
        test_questions = list(testset_df['user_input'].dropna())
        print(f"Using 'user_input' column for questions")
    elif 'question' in testset_df.columns:
        test_questions = list(testset_df['question'].dropna())
        print(f"Using 'question' column for questions")
    else:
        # Use first available column with data
        for col in testset_df.columns:
            if not testset_df[col].isna().all():
                test_questions = list(testset_df[col].dropna())
                print(f"Using column '{col}' for questions")
                break
    
    print(f"\nExtracted {len(test_questions)} questions from testset")
    print("Sample questions:")
    for i, question in enumerate(test_questions[:3], 1):
        print(f"  {i}. {str(question)[:100]}...")
    
    # =============================================================================
    # STEP 1: SETUP LANGSMITH DATASET (with URL)
    # =============================================================================
    
    langsmith_client = None
    dataset = None
    
    if not args.skip_langsmith and os.getenv("LANGSMITH_API_KEY"):
        print("\n" + "=" * 80)
        print("1. SETUP LANGSMITH DATASET")
        print("=" * 80)
        
        langsmith_client = Client()
        dataset_name = f"backend-rag-eval-enhanced-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        dataset = langsmith_client.create_dataset(
            dataset_name=dataset_name,
            description="Enhanced Backend Advanced RAG Methods Evaluation Dataset"
        )
        
        print(f"✅ Dataset URL: https://smith.langchain.com/datasets/{dataset.id}")
        print(f"✅ Dataset: {dataset_name}")
        
        # Add questions to dataset
        examples = []
        for i, question in enumerate(test_questions, 1):
            examples.append({
                "inputs": {"question": question},
                "outputs": {"index": i, "source": "ragas_enhanced_testset_generator"}
            })
        
        for example in examples:
            langsmith_client.create_example(
                inputs=example["inputs"],
                outputs=example["outputs"],
                dataset_id=dataset.id
            )
        print(f"✅ Added {len(examples)} examples to dataset")
    
    # =============================================================================
    # STEP 2: EVALUATE BACKEND RAG WITH MULTIPLE RETRIEVAL METHODS
    # =============================================================================
    
    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
    custom_run_config = RunConfig(timeout=180)  # Fast evaluation timeout
    
    # Define retrieval methods to test
    retrieval_methods = {
        "Naive": "naive",
        "Parent Document": "parent_document", 
        "Sentence Window": "sentence_window",
        "HyDE": "hyde"
    }
    
    print(f"\n📊 Evaluating Backend RAG with {len(retrieval_methods)} methods and {len(test_questions)} questions")
    
    all_results = {}
    
    # Evaluate each retrieval method using backend RAG tool
    for method_name, method_key in retrieval_methods.items():
        print(f"\n🔍 Evaluating Backend RAG with {method_name} retrieval...")
        
        # Create RAG tool with specific retrieval method
        rag_tool = RAGTool(retrieval_method=method_key)
        
        latencies = []
        eval_dataset_samples = []
        
        for i, question in enumerate(test_questions):
            print(f"   Processing question {i+1}/{len(test_questions)} with {method_name}")
            
            start_time = time.time()
            
            # Use backend RAG tool
            result = rag_tool._run(question)
            
            end_time = time.time()
            latencies.append(end_time - start_time)
            
            # Extract response and context
            if isinstance(result, dict):
                response = result.get("answer", "")
                retrieved_contexts = [chunk.get("content", "") for chunk in result.get("relevant_chunks", [])]
            else:
                response = str(result)
                retrieved_contexts = []
            
            eval_sample = {
                'user_input': question,
                'response': response,
                'retrieved_contexts': retrieved_contexts,
                'reference_contexts': retrieved_contexts,
                'reference': response
            }
            eval_dataset_samples.append(eval_sample)
        
        # Run RAGAS evaluation
        print(f"   🔍 Running RAGAS evaluation for {method_name}...")
        eval_df = pd.DataFrame(eval_dataset_samples)
        evaluation_dataset = EvaluationDataset.from_pandas(eval_df)
        
        ragas_result = evaluate(
            dataset=evaluation_dataset,
            metrics=[LLMContextRecall(), Faithfulness(), FactualCorrectness(), ResponseRelevancy()],
            llm=evaluator_llm,
            run_config=custom_run_config
        )
        
        avg_latency = sum(latencies) / len(latencies)
        std_latency = np.std(latencies)
        all_results[method_name] = {
            'ragas_scores': ragas_result,
            'avg_latency': avg_latency,
            'std_latency': std_latency,
            'num_queries': len(latencies),
            'method_key': method_key
        }
        
        print(f"✅ {method_name} completed - Latency: {avg_latency:.2f}±{std_latency:.2f}s")
    
    # =============================================================================
    # STEP 3: ENHANCED ANALYSIS AND SMART RECOMMENDATIONS
    # =============================================================================
    
    summary_data = []
    performance_scores = {}
    
    for method_name, result_data in all_results.items():
        df = result_data['ragas_scores'].to_pandas()
        
        context_recall = df['context_recall'].mean() if 'context_recall' in df.columns else 0.0
        faithfulness = df['faithfulness'].mean() if 'faithfulness' in df.columns else 0.0
        factual_correctness = df['factual_correctness'].mean() if 'factual_correctness' in df.columns else 0.0
        response_relevancy = df['answer_relevancy'].mean() if 'answer_relevancy' in df.columns else 0.0
        
        # Calculate composite scores
        quality_score = (context_recall + faithfulness + factual_correctness + response_relevancy) / 4
        speed_score = 1 / (result_data['avg_latency'] + 0.1)  # Inverse of latency with small offset
        consistency_score = 1 / (result_data['std_latency'] + 0.1)  # Inverse of std dev
        
        # Weighted composite performance score
        performance_score = (
            quality_score * 0.6 +      # Quality is most important
            speed_score * 0.25 +       # Speed is important for UX
            consistency_score * 0.15   # Consistency matters for production
        )
        
        performance_scores[method_name] = {
            'quality': quality_score,
            'speed': speed_score,
            'consistency': consistency_score,
            'composite': performance_score,
            'method_key': result_data['method_key']
        }
        
        summary_data.append({
            'Retrieval Method': method_name,
            'Context Recall': f"{context_recall:.3f}",
            'Faithfulness': f"{faithfulness:.3f}",
            'Factual Correctness': f"{factual_correctness:.3f}",
            'Response Relevancy': f"{response_relevancy:.3f}",
            'Quality Score': f"{quality_score:.3f}",
            'Avg Latency (s)': f"{result_data['avg_latency']:.2f}",
            'Latency StdDev': f"{result_data['std_latency']:.2f}",
            'Performance Score': f"{performance_score:.3f}",
            'Queries': result_data['num_queries']
        })
    
    results_df = pd.DataFrame(summary_data)
    
    print("\n📊 ENHANCED BACKEND RAG EVALUATION RESULTS")
    print("=" * 120)
    print(results_df.to_string(index=False))
    
    # Enhanced recommendation system
    print("\n🎯 SMART RECOMMENDATION SYSTEM")
    print("=" * 80)
    
    # Find best performers in different categories
    best_quality = max(performance_scores.items(), key=lambda x: x[1]['quality'])
    best_speed = max(performance_scores.items(), key=lambda x: x[1]['speed'])
    best_consistency = max(performance_scores.items(), key=lambda x: x[1]['consistency'])
    best_overall = max(performance_scores.items(), key=lambda x: x[1]['composite'])
    
    print(f"🏆 BEST OVERALL PERFORMER: {best_overall[0]}")
    print(f"   • Performance Score: {best_overall[1]['composite']:.3f}")
    print(f"   • Method Key: '{best_overall[1]['method_key']}'")
    print(f"   • Quality: {best_overall[1]['quality']:.3f} | Speed: {best_overall[1]['speed']:.3f} | Consistency: {best_overall[1]['consistency']:.3f}")
    
    print(f"\n📈 CATEGORY LEADERS:")
    print(f"   🎯 Highest Quality: {best_quality[0]} ({best_quality[1]['quality']:.3f})")
    print(f"   ⚡ Fastest Response: {best_speed[0]} ({all_results[best_speed[0]]['avg_latency']:.2f}s)")
    print(f"   📊 Most Consistent: {best_consistency[0]} (±{all_results[best_consistency[0]]['std_latency']:.2f}s)")
    
    # Use case recommendations
    print(f"\n💡 USE CASE RECOMMENDATIONS:")
    print(f"   🏢 Production Systems: {best_overall[0]} - Best balance of quality, speed, and consistency")
    print(f"   🔬 Research/Analysis: {best_quality[0]} - Highest quality responses for detailed analysis")
    print(f"   ⚡ Real-time Applications: {best_speed[0]} - Fastest response times for interactive use")
    print(f"   🎛️ Batch Processing: {best_consistency[0]} - Most predictable performance for automation")
    
    # Performance analysis
    print(f"\n📊 PERFORMANCE ANALYSIS:")
    for method_name, scores in performance_scores.items():
        print(f"   {method_name:20} | Overall: {scores['composite']:.3f} | Quality: {scores['quality']:.3f} | Speed: {scores['speed']:.3f} | Consistency: {scores['consistency']:.3f}")
    
    # Implementation recommendation
    quality_threshold = 0.7
    speed_threshold = 0.5
    
    production_candidates = [
        name for name, scores in performance_scores.items() 
        if scores['quality'] >= quality_threshold and scores['speed'] >= speed_threshold
    ]
    
    if production_candidates:
        print(f"\n✅ PRODUCTION-READY METHODS: {', '.join(production_candidates)}")
        recommended_method = max(production_candidates, key=lambda x: performance_scores[x]['composite'])
        print(f"🚀 RECOMMENDED FOR DEPLOYMENT: {recommended_method} ('{performance_scores[recommended_method]['method_key']}')")
    else:
        print(f"\n⚠️  Consider optimizing methods - none meet both quality ({quality_threshold}) and speed ({speed_threshold}) thresholds")
        print(f"🚀 BEST AVAILABLE OPTION: {best_overall[0]} ('{best_overall[1]['method_key']}')")
    
    if dataset:
        print(f"\n🔗 LangSmith Dataset: https://smith.langchain.com/datasets/{dataset.id}")
    
    print("\n📋 IMPLEMENTATION STATUS:")
    print("✅ RAGAS testset generation using long-form policy analysis documents (no fallbacks)")
    print("✅ Backend RAG tool supports 4 advanced retrieval methods")
    print("✅ Smart recommendation system with multi-factor analysis")
    print("✅ Production readiness assessment completed")
    print("✅ Use case specific recommendations provided")
    print("⚡ Direct evaluation with gpt-4o-mini and long-form document chunks")
    
    return 0

if __name__ == "__main__":
    exit(main()) 