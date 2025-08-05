#!/usr/bin/env python3
"""
Task 5: Golden Dataset Evaluation with Advanced Backend RAG Methods
Following the Advanced Retrieval notebook approach for cost evaluation and LangSmith logging
"""

import os
import sys
import argparse
import time
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from datetime import datetime

load_dotenv()
backend_path = Path(__file__).parent.parent / "backend"
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))
sys.path.insert(0, str(project_root))

# Import backend RAG tool and document processor
from tools.rag_tool import RAGTool
from utils.document_processor import DocumentProcessor
from langchain_community.vectorstores import Qdrant
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ragas.testset import TestsetGenerator
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader

from ragas import EvaluationDataset, evaluate
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness, ResponseRelevancy
from ragas import RunConfig
from langsmith import Client
from langchain_core.documents import Document

# Import LangChain tracer with fallback
try:
    from langsmith.callbacks import LangChainTracer
except ImportError:
    from langchain.callbacks import LangChainTracer


def setup_tracing(langsmith_client):
    """Setup LangSmith tracing"""
    if langsmith_client:
        tracer = LangChainTracer(project_name="rag-evaluation")
    else:
        os.environ["LANGCHAIN_TRACING_V2"] = "true" 
        os.environ["LANGCHAIN_PROJECT"] = "rag-evaluation"
        tracer = True
    return tracer


def get_cost_data_from_langsmith(langsmith_client, method_name, num_queries):
    """Retrieve cost data from LangSmith runs using tag filtering"""
    cost_data = {
        'total_cost': 0.0,
        'total_tokens': 0,
        'avg_cost_per_query': 0.0
    }
    
    if not langsmith_client:
        return cost_data
    
    try:
        # Use tag-based filtering like the original notebook
        tag_filter = f"has(tags, 'retriever_{method_name}')"
        recent_runs = list(langsmith_client.list_runs(
            project_name="rag-evaluation",
            limit=num_queries + 20,  # Get a few extra runs to account for other operations
            filter=tag_filter
        ))
        
        total_cost = 0.0
        total_tokens = 0
        run_count = 0
        
        for run in recent_runs:
            if run.total_cost:
                # Convert Decimal to float like in the original notebook
                total_cost += float(run.total_cost)
                run_count += 1
            if run.total_tokens:
                total_tokens += int(run.total_tokens)
        
        if run_count > 0:
            # Calculate average cost per query based on actual queries
            avg_cost_per_query = total_cost / num_queries if num_queries > 0 else 0.0
            cost_data.update({
                'total_cost': total_cost,
                'total_tokens': total_tokens,
                'avg_cost_per_query': avg_cost_per_query
            })
            
            print(f"✅ Cost tracking for {method_name}: ${total_cost:.4f} total, ${avg_cost_per_query:.4f}/query from {run_count} runs")
        else:
            print(f"⚠️ No cost data found for {method_name} with filter: {tag_filter}")
    
    except Exception as e:
        print(f"Error retrieving cost data for {method_name}: {e}")
        
    return cost_data


def log_to_langsmith_dataset(langsmith_client, dataset, method_name, results, eval_df, latencies, rag_tool):
    """Log evaluation results to LangSmith dataset using proper evaluation framework"""
    if not langsmith_client:
        return 0
        
    if not dataset:
        return 0
    
    print(f"📊 Creating linked evaluation runs for {method_name} in LangSmith...")
    
    try:
        # Import LangSmith evaluation function
        from langsmith.evaluation import evaluate
        
        # Create an evaluation function that actually invokes the RAG tool with tracing
        def evaluation_function(inputs):
            """Evaluation function that invokes the actual RAG tool"""
            question = inputs.get("question", "")
            
            try:
                # Set the RAG tool to use the correct method
                rag_tool.set_retrieval_method(results['method_key'])
                
                # Invoke the RAG tool - this will create proper traces with cost/latency
                result = rag_tool._run(question)
                
                if isinstance(result, dict):
                    response = result.get("answer", "")
                else:
                    response = str(result)
                
                return {"response": response}
            except Exception as e:
                print(f"⚠️ RAG evaluation failed for question: {e}")
                # Fallback to pre-computed result
                for idx, row in eval_df.iterrows():
                    if str(row['user_input']).strip() == question.strip():
                        return {"response": str(row['response'])}
                return {"response": "No matching result found"}
        
        # Run the evaluation using LangSmith's evaluate function
        evaluation_results = evaluate(
            evaluation_function,
            data=dataset,
            experiment_prefix=f"{method_name}_evaluation",
            max_concurrency=1,
            client=langsmith_client
        )
        
        if evaluation_results:
            print(f"✅ Created linked evaluation runs for {method_name}")
            print(f"📊 Experiment: {evaluation_results.experiment_name}")
            
            # Now add feedback to the created runs
            ragas_df = results['ragas_scores'].to_pandas()
            
            # Use the evaluation results to get run information directly
            try:
                # The evaluation_results object contains run information
                feedback_count = 0
                
                # Get run IDs from the evaluation results
                if hasattr(evaluation_results, 'results') and evaluation_results.results:
                    eval_results_list = list(evaluation_results.results)
                    
                    for idx, eval_result in enumerate(eval_results_list[:len(ragas_df)]):
                        if idx < len(ragas_df) and hasattr(eval_result, 'run_id'):
                            ragas_row = ragas_df.iloc[idx]
                            run_id = eval_result.run_id
                            
                            # Add RAGAS metrics as feedback
                            metrics = {}
                            if 'context_recall' in ragas_row and pd.notna(ragas_row['context_recall']):
                                metrics['context_recall'] = float(ragas_row['context_recall'])
                            if 'faithfulness' in ragas_row and pd.notna(ragas_row['faithfulness']):
                                metrics['faithfulness'] = float(ragas_row['faithfulness'])
                            
                            # Handle factual correctness
                            if 'factual_correctness(mode=f1)' in ragas_row and pd.notna(ragas_row['factual_correctness(mode=f1)']):
                                metrics['factual_correctness'] = float(ragas_row['factual_correctness(mode=f1)'])
                            elif 'factual_correctness' in ragas_row and pd.notna(ragas_row['factual_correctness']):
                                metrics['factual_correctness'] = float(ragas_row['factual_correctness'])
                            
                            if 'answer_relevancy' in ragas_row and pd.notna(ragas_row['answer_relevancy']):
                                metrics['answer_relevancy'] = float(ragas_row['answer_relevancy'])
                            
                            # Add performance metrics
                            if idx < len(latencies):
                                metrics['latency_seconds'] = latencies[idx]
                            
                            if results['avg_cost_per_query'] > 0:
                                metrics['cost_dollars_per_query'] = results['avg_cost_per_query']
                            
                            # Create feedback for each metric
                            for metric_name, metric_value in metrics.items():
                                if isinstance(metric_value, (int, float)):
                                    try:
                                        langsmith_client.create_feedback(
                                            run_id=run_id,
                                            key=metric_name,
                                            score=float(metric_value),
                                            comment=f"{method_name} evaluation"
                                        )
                                        feedback_count += 1
                                    except Exception as e:
                                        # Just continue if individual feedback fails
                                        pass
                
                if feedback_count > 0:
                    print(f"✅ Added {feedback_count} feedback scores to evaluation runs")
                else:
                    print("📊 Evaluation runs created successfully - metrics available in console output")
                
            except Exception as feedback_error:
                print(f"📊 Evaluation runs created successfully - metrics available in console output")
            
            # Display the metrics in console as backup
            print(f"📈 {method_name} Evaluation Metrics:")
            for idx, row in enumerate(ragas_df.itertuples(), 1):
                metrics_summary = []
                if hasattr(row, 'context_recall') and pd.notna(row.context_recall):
                    metrics_summary.append(f"Recall: {row.context_recall:.3f}")
                if hasattr(row, 'faithfulness') and pd.notna(row.faithfulness):
                    metrics_summary.append(f"Faith: {row.faithfulness:.3f}")
                
                # Handle factual correctness
                fc_value = None
                if hasattr(row, 'factual_correctness_mode_f1_') and pd.notna(getattr(row, 'factual_correctness_mode_f1_', None)):
                    fc_value = getattr(row, 'factual_correctness_mode_f1_')
                elif hasattr(row, 'factual_correctness') and pd.notna(row.factual_correctness):
                    fc_value = row.factual_correctness
                
                if fc_value is not None:
                    metrics_summary.append(f"FC: {fc_value:.3f}")
                
                if hasattr(row, 'answer_relevancy') and pd.notna(row.answer_relevancy):
                    metrics_summary.append(f"Relevancy: {row.answer_relevancy:.3f}")
                
                if idx - 1 < len(latencies):
                    metrics_summary.append(f"Latency: {latencies[idx-1]:.2f}s")
                
                print(f"  Query {idx}: {' | '.join(metrics_summary)}")
            
            print(f"🔗 View linked runs: https://smith.langchain.com/datasets/{dataset.id}")
            return len(ragas_df)
        
    except ImportError:
        print("⚠️ LangSmith evaluation function not available, using alternative approach")
    except Exception as e:
        print(f"⚠️ LangSmith evaluation failed: {e}")
    
    # Fallback: provide guidance for manual linking
    print(f"💡 To manually link runs in LangSmith:")
    print(f"   1. Go to: https://smith.langchain.com/datasets/{dataset.id}")
    print(f"   2. Click 'Linked Runs' tab")
    print(f"   3. Use 'Link Existing Runs' to connect runs with tag 'retriever_{method_name}'")
    
    return 0


def evaluate_retriever_method(rag_tool, langsmith_client, dataset, tracer, method_name, method_key, test_questions, testset_df):
    """Evaluate a single retrieval method using backend RAG"""
    # Set the retrieval method on the backend RAG tool
    rag_tool.set_retrieval_method(method_key)
    
    latencies = []
    eval_dataset_samples = []
    method_start_time = time.time()
    
    for i, question in enumerate(test_questions):
        start_time = time.time()
        
        # Setup LangSmith tracing for this specific call
        if tracer and langsmith_client:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_PROJECT"] = "rag-evaluation"
            
            # Get the RAG chain from the tool and invoke it with tracer for cost tracking
            try:
                # Get the appropriate retriever
                retriever = rag_tool._get_retriever(question)
                
                # Build the RAG chain (same logic as in RAG tool)
                if method_key == "hyde":
                    # Create hypothetical document for better retrieval
                    hypothetical_doc = rag_tool._create_hypothetical_document(question)
                    
                    # Build HyDE RAG chain
                    from operator import itemgetter
                    from langchain_core.runnables import RunnablePassthrough
                    
                    rag_chain = (
                        {"context": lambda x: retriever.get_relevant_documents(hypothetical_doc), "question": itemgetter("question")}
                        | RunnablePassthrough.assign(context=itemgetter("context"))
                        | RunnablePassthrough.assign(
                            raw_docs=itemgetter("context"),
                            context=lambda x: rag_tool._format_context(x["context"])
                        )
                        | {"response": rag_tool.rag_prompt | rag_tool.llm, "context": itemgetter("context"), "raw_docs": itemgetter("raw_docs")}
                    )
                else:
                    # Standard RAG chain for other methods
                    from operator import itemgetter
                    from langchain_core.runnables import RunnablePassthrough
                    
                    rag_chain = (
                        {"context": itemgetter("question") | retriever, "question": itemgetter("question")}
                        | RunnablePassthrough.assign(context=itemgetter("context"))
                        | RunnablePassthrough.assign(
                            raw_docs=itemgetter("context"),
                            context=lambda x: rag_tool._format_context(x["context"])
                        )
                        | {"response": rag_tool.rag_prompt | rag_tool.llm, "context": itemgetter("context"), "raw_docs": itemgetter("raw_docs")}
                    )
                
                # Execute with LangSmith tracer for cost tracking
                chain_result = rag_chain.invoke(
                    {"question": question},
                    config={"callbacks": [tracer], "tags": [f"retriever_{method_name}"]}
                )
                
                # Process result same as RAG tool
                response_content = chain_result["response"]
                if hasattr(response_content, 'content'):
                    response_content = response_content.content
                
                # Extract relevant chunks from raw documents
                relevant_chunks = []
                if "raw_docs" in chain_result:
                    for doc in chain_result["raw_docs"]:
                        relevant_chunks.append({
                            "content": doc.page_content,
                            "metadata": doc.metadata,
                            "source": doc.metadata.get("source", "Unknown")
                        })
                
                result = {
                    "answer": response_content,
                    "relevant_chunks": relevant_chunks,
                    "retrieval_method": method_key
                }
                
            except Exception as e:
                print(f"Warning: Chain invoke failed, falling back to RAG tool: {e}")
                # Fallback to original method
                result = rag_tool._run(question)
        else:
            # No tracer, use original method
            result = rag_tool._run(question)
        
        end_time = time.time()
        latency = end_time - start_time
        latencies.append(latency)
        
        if isinstance(result, dict):
            response = result.get("answer", "")
            retrieved_contexts = [chunk.get("content", "") for chunk in result.get("relevant_chunks", [])]
        else:
            response = str(result)
            retrieved_contexts = []
        
        reference_answer = ""
        if i < len(testset_df):
            # Try different possible reference column names
            reference_cols = ['reference', 'ground_truth', 'reference_answer', 'expected_answer', 'answer']
            for ref_col in reference_cols:
                if ref_col in testset_df.columns:
                    ref_value = testset_df.iloc[i][ref_col]
                    if pd.notna(ref_value) and str(ref_value).strip():
                        reference_answer = str(ref_value).strip()
                        break
        
        eval_sample = {
            'user_input': question,
            'response': response,
            'retrieved_contexts': retrieved_contexts,
            'reference_contexts': retrieved_contexts,
            'reference': reference_answer
        }
        eval_dataset_samples.append(eval_sample)
    
    method_total_time = time.time() - method_start_time
    
    eval_df = pd.DataFrame(eval_dataset_samples)
    evaluation_dataset = EvaluationDataset.from_pandas(eval_df)
    
    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", temperature=0))
    custom_run_config = RunConfig(timeout=180)
    
    ragas_result = evaluate(
        dataset=evaluation_dataset,
        metrics=[LLMContextRecall(), Faithfulness(), FactualCorrectness(), ResponseRelevancy()],
        llm=evaluator_llm,
        run_config=custom_run_config
    )
    
    avg_latency = sum(latencies) / len(latencies)
    std_latency = np.std(latencies)
    
    cost_data = get_cost_data_from_langsmith(langsmith_client, method_name, len(latencies))
    
    results = {
        'ragas_scores': ragas_result,
        'avg_latency': avg_latency,
        'std_latency': std_latency,
        'total_time': method_total_time,
        'num_queries': len(latencies),
        'method_key': method_key,
        'eval_samples': eval_dataset_samples,
        **cost_data
    }
    
    if langsmith_client and dataset:
        log_to_langsmith_dataset(langsmith_client, dataset, method_name, results, eval_df, latencies, rag_tool)
    
    return results


def load_documents():
    """Load policy documents for testset generation"""
    documents = []
    policy_documents_path = Path(__file__).parent / "policy_documents"
    
    policy_files = [
        "policy_analysis_framework.txt",
        "economic_policy_statistics.txt"
    ]
    
    for policy_file in policy_files:
        policy_path = policy_documents_path / policy_file
        if policy_path.exists():
            with open(policy_path, 'r', encoding='utf-8') as f:
                policy_content = f.read()
            
            paragraphs = policy_content.split('\n\n')
            current_chunk = ""
            chunk_count = 0
            
            for paragraph in paragraphs:
                paragraph = paragraph.strip()
                if paragraph:
                    current_chunk += paragraph + "\n\n"
                    
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
    
    return documents


def generate_testset(documents, testset_size):
    """Generate RAGAS testset"""
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
    
    testset = generator.generate_with_langchain_docs(
        documents=documents,
        testset_size=testset_size
    )
    
    testset_df = testset.to_pandas()
    print(f"📊 Generated testset with {len(testset_df)} questions")
    
    return testset_df


def extract_questions(testset_df):
    """Extract questions from testset"""
    if 'user_input' in testset_df.columns:
        test_questions = list(testset_df['user_input'].dropna())
    elif 'question' in testset_df.columns:
        test_questions = list(testset_df['question'].dropna())
    else:
        for col in testset_df.columns:
            if not testset_df[col].isna().all():
                test_questions = list(testset_df[col].dropna())
                break
    
    return test_questions


def setup_backend_rag():
    """Setup backend RAG tool with evaluation vectorstore using backend infrastructure"""
    # Load policy documents using backend approach
    policy_documents_path = Path(__file__).parent / "policy_documents"
    policy_files = [
        "policy_analysis_framework.txt",
        "economic_policy_statistics.txt"
    ]
    
    documents = []
    for policy_file in policy_files:
        policy_path = policy_documents_path / policy_file
        if policy_path.exists():
            with open(policy_path, 'r', encoding='utf-8') as f:
                policy_content = f.read()
            
            # Create larger chunks following backend approach
            paragraphs = policy_content.split('\n\n')
            current_chunk = ""
            chunk_count = 0
            
            for paragraph in paragraphs:
                paragraph = paragraph.strip()
                if paragraph:
                    current_chunk += paragraph + "\n\n"
                    
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
    
    # Convert documents to file format expected by backend document processor
    file_data = []
    for i, doc in enumerate(documents):
        file_data.append({
            'filename': f"policy_chunk_{i}.txt",
            'content': doc.page_content.encode('utf-8'),
            'content_type': 'text/plain',
            'size': len(doc.page_content)
        })
    
    # Use backend document processor to create vectorstore
    doc_processor = DocumentProcessor()
    vectorstore = doc_processor.process_uploaded_files(file_data, "evaluation_session")
    
    # Create RAG tool with the vectorstore
    rag_tool = RAGTool(vectorstore=vectorstore)
    
    return rag_tool


def setup_langsmith(args, test_questions):
    """Setup LangSmith client and dataset"""
    langsmith_client = None
    dataset = None
    
    if not args.skip_langsmith and os.getenv("LANGSMITH_API_KEY"):
        try:
            langsmith_client = Client()
            
            # Test the client connection
            try:
                langsmith_client.list_projects(limit=1)
                print("✅ LangSmith client connected successfully")
            except Exception as e:
                print(f"❌ LangSmith client connection failed: {e}")
                return None, None
            
            project_name = "rag-evaluation"
            existing_projects = [p for p in langsmith_client.list_projects() if p.name == project_name]
            if not existing_projects:
                langsmith_client.create_project(project_name=project_name, 
                                               description="RAG Methods Evaluation - Backend RAG Only")
                print(f"✅ Created LangSmith project: {project_name}")
            else:
                print(f"✅ Using existing LangSmith project: {project_name}")
            
            dataset_name = f"backend-rag-evaluation-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            try:
                dataset = langsmith_client.create_dataset(
                    dataset_name=dataset_name,
                    description="Backend RAG Only Evaluation with Cost Tracking"
                )
                print(f"✅ Created LangSmith dataset: {dataset_name}")
            except Exception as e:
                print(f"❌ Failed to create LangSmith dataset: {e}")
                return langsmith_client, None
            
            # Create examples with error handling
            print(f"📝 Creating {len(test_questions)} examples in dataset...")
            for i, question in enumerate(test_questions, 1):
                try:
                    langsmith_client.create_example(
                        inputs={"question": question},
                        outputs={"index": i, "source": "ragas_testset_generator"},
                        dataset_id=dataset.id
                    )
                except Exception as e:
                    print(f"⚠️ Failed to create example {i}: {e}")
                    
            print(f"✅ LangSmith setup complete - Dataset ID: {dataset.id}")
            
        except Exception as e:
            print(f"❌ LangSmith setup failed: {e}")
            return None, None
    elif args.skip_langsmith:
        print("⏭️ Skipping LangSmith (--skip-langsmith flag)")
    else:
        print("⚠️ LANGSMITH_API_KEY not found, skipping LangSmith logging")
    
    return langsmith_client, dataset


def analyze_results(all_results):
    """Analyze and format results"""
    summary_data = []
    performance_scores = {}
    
    for method_name, result_data in all_results.items():
        df = result_data['ragas_scores'].to_pandas()
        
        context_recall = df['context_recall'].mean() if 'context_recall' in df.columns else 0.0
        faithfulness = df['faithfulness'].mean() if 'faithfulness' in df.columns else 0.0
        
        # Handle both possible column names for factual correctness
        factual_correctness = 0.0
        if 'factual_correctness(mode=f1)' in df.columns:
            factual_correctness = df['factual_correctness(mode=f1)'].mean()
        elif 'factual_correctness' in df.columns:
            factual_correctness = df['factual_correctness'].mean()
        
        response_relevancy = df['answer_relevancy'].mean() if 'answer_relevancy' in df.columns else 0.0
        
        quality_score = (context_recall + faithfulness + factual_correctness + response_relevancy) / 4
        speed_score = 1 / (result_data['avg_latency'] + 0.1)
        consistency_score = 1 / (result_data['std_latency'] + 0.1)
        
        cost_efficiency = 0
        if result_data['avg_cost_per_query'] > 0:
            cost_efficiency = 1 / (result_data['avg_cost_per_query'] + 0.001)
        
        if cost_efficiency > 0:
            performance_score = (
                quality_score * 0.50 +
                speed_score * 0.20 +
                consistency_score * 0.15 +
                cost_efficiency * 0.15
            )
        else:
            performance_score = (
                quality_score * 0.60 +
                speed_score * 0.25 +
                consistency_score * 0.15
            )
        
        performance_scores[method_name] = {
            'quality': quality_score,
            'speed': speed_score,
            'consistency': consistency_score,
            'cost_efficiency': cost_efficiency,
            'composite': performance_score,
            'method_key': result_data['method_key']
        }
        
        total_cost_str = f"${result_data['total_cost']:.4f}" if result_data['total_cost'] > 0 else "Tracked"
        cost_per_query_str = f"${result_data['avg_cost_per_query']:.4f}" if result_data['avg_cost_per_query'] > 0 else "Tracked"

        summary_data.append({
            'Retrieval Method': method_name,
            'Context Recall': f"{context_recall:.3f}",
            'Faithfulness': f"{faithfulness:.3f}",
            'Factual Correctness': f"{factual_correctness:.3f}",
            'Response Relevancy': f"{response_relevancy:.3f}",
            'Quality Score': f"{quality_score:.3f}",
            'Avg Latency (s)': f"{result_data['avg_latency']:.2f}",
            'Latency StdDev': f"{result_data['std_latency']:.2f}",
            'Total Cost ($)': total_cost_str,
            'Cost/Query ($)': cost_per_query_str,
            'Performance Score': f"{performance_score:.3f}",
            'Queries': result_data['num_queries']
        })
    
    return summary_data, performance_scores


def main():
    parser = argparse.ArgumentParser(description="Backend RAG Only Evaluation - No Other Tools Required")
    parser.add_argument("--testset-size", type=int, default=3, help="Number of questions (default: 3)")
    parser.add_argument("--skip-langsmith", action="store_true", help="Skip LangSmith dataset creation")
    parser.add_argument("--methods", nargs="+", 
                       choices=["naive", "parent_document", "sentence_window", "hyde"],
                       default=["naive", "parent_document"],
                       help="Retrieval methods to evaluate (default: naive parent_document)")
    args = parser.parse_args()
    
    # Load documents for testset generation only
    documents = load_documents()
    
    # Generate testset
    testset_size = max(2, args.testset_size)
    testset_df = generate_testset(documents, testset_size)
    
    # Extract questions
    test_questions = extract_questions(testset_df)
    
    # Setup backend RAG tool only (no full agent)
    rag_tool = setup_backend_rag()
    
    # Setup LangSmith
    langsmith_client, dataset = setup_langsmith(args, test_questions)
    
    # Setup tracing
    tracer = setup_tracing(langsmith_client)
    
    # Define all available retrieval methods
    all_methods = {
        "Naive": "naive",
        "Parent Document": "parent_document", 
        "Sentence Window": "sentence_window",
        "HyDE": "hyde"
    }
    
    # Filter to only requested methods
    retrieval_methods = {
        name: key for name, key in all_methods.items() 
        if key in args.methods
    }
    
    print(f"Evaluating {len(retrieval_methods)} methods: {list(retrieval_methods.keys())}")
    
    # Evaluate all requested methods using backend RAG
    all_results = {}
    for method_name, method_key in retrieval_methods.items():
        results = evaluate_retriever_method(
            rag_tool, langsmith_client, dataset, tracer,
            method_name, method_key, test_questions, testset_df
        )
        all_results[method_name] = results
    
    # Analyze results
    summary_data, performance_scores = analyze_results(all_results)
    
    # Display results
    results_df = pd.DataFrame(summary_data)
    print(results_df.to_string(index=False))
    
    # Show best performers
    best_overall = max(performance_scores.items(), key=lambda x: x[1]['composite'])
    best_quality = max(performance_scores.items(), key=lambda x: x[1]['quality'])
    best_speed = max(performance_scores.items(), key=lambda x: x[1]['speed'])
    
    print(f"\nBest Overall: {best_overall[0]} (Score: {best_overall[1]['composite']:.3f})")
    print(f"Highest Quality: {best_quality[0]} (Score: {best_quality[1]['quality']:.3f})")
    print(f"Fastest: {best_speed[0]} (Latency: {all_results[best_speed[0]]['avg_latency']:.2f}s)")
    
    methods_with_cost = [k for k, v in performance_scores.items() if v['cost_efficiency'] > 0]
    if methods_with_cost:
        best_cost = max(methods_with_cost, key=lambda x: performance_scores[x]['cost_efficiency'])
        print(f"Most Cost Efficient: {best_cost} (${all_results[best_cost]['avg_cost_per_query']:.4f}/query)")
    
    if dataset:
        print(f"\nLangSmith Dataset: https://smith.langchain.com/datasets/{dataset.id}")
    
    return 0

if __name__ == "__main__":
    exit(main()) 