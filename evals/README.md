# Policy Analyzer - Golden Dataset Evaluation (Task 5)

This directory implements **Task 5: Creating a Golden Test Data Set** using a **single consolidated file** with RAGAS TestsetGenerator and full backend Qdrant integration.

## 🎯 **Single File + Backend-Only Implementation**

**Complete evaluation in a single file that REQUIRES backend integration:**
- **`golden_dataset_evaluation.py`** - Complete evaluation pipeline (backend-only)
- **`setup_evaluation.py`** - Environment validation with backend requirements
- **`pyproject.toml`** - Dependencies with uv

## Overview

The evaluation system **requires backend integration** and implements:

1. **RAGAS TestsetGenerator**: Using the exact LangchainLLMWrapper approach from notebook  
2. **Backend Qdrant Only**: **Always uses backend Qdrant - no fallbacks**
3. **Backend RAG Tool Only**: **Always uses backend RAG tool - no standalone mode**
4. **Backend Document Processor**: **Always uses backend document processing**
5. **LangSmith Dataset Setup**: Optional LangSmith integration for tracking
6. **Baseline RAGAS Assessment**: Evaluating against Task 5 expected metrics

> ⚠️ **Backend Required**: This implementation fails fast if backend components cannot be imported. No fallback implementations.

## Key Features ✨

### 🎯 RAGAS TestsetGenerator (Pandas Pattern)
```python
from ragas.testset import TestsetGenerator
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

generator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"))
generator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

generator = TestsetGenerator(llm=generator_llm, embedding_model=generator_embeddings)
testset = generator.generate_with_langchain_docs(sample_docs, testset_size=20)

# Extract questions using pandas pattern
test_questions = list(testset.to_pandas()['user_input'])
print(f" Generated {len(test_questions)} questions")
```

### 🔧 Backend Infrastructure Integration
- **RAG Tool**: Uses existing `backend/tools/rag_tool.py`
- **Document Processing**: Leverages `backend/utils/document_processor.py`
- **Qdrant Database**: **Connects to and reuses backend Qdrant collections**
- **Configuration**: Inherits all backend settings and environment variables
- **Session Management**: Uses backend session management approach

### 📊 Expected Results (Task 5 Baseline)
| Metric | Expected Score | Implementation Status |
|--------|---------------|---------------------|
| `context_precision` | 0.85 | ✅ Fully implemented |
| `context_recall` | 0.81 | ✅ Key improvement area identified |
| `faithfulness` | 0.87 | ✅ Fully implemented |
| `answer_relevancy` | 0.92 | ✅ Fully implemented |

## Quick Start 🚀

### 1. Setup Environment (Backend Required)
```bash
cd evals

# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Validate setup and install dependencies (REQUIRES backend)
uv run setup_evaluation.py

# If backend integration fails, setup backend first:
# cd ../backend && uv sync
```

### 2. Configure API Keys
```bash
# Edit the generated .env file
nano .env

# Add your keys:
OPENAI_API_KEY=your_openai_api_key_here
LANGCHAIN_API_KEY=your_langchain_api_key_here  # For LangSmith
```

### 3. Run Complete Evaluation
```bash
# Full evaluation pipeline (single command!)
uv run golden_dataset_evaluation.py

# Custom options
uv run golden_dataset_evaluation.py --testset-size 30
uv run golden_dataset_evaluation.py --skip-langsmith
uv run golden_dataset_evaluation.py --langsmith-interactive
```

## Backend-Only Architecture

### ✅ **Always Uses Backend Infrastructure**
The evaluation system **REQUIRES backend integration** and:

1. **Backend Qdrant Only**: Uses backend's exact Qdrant configuration and location
2. **Backend Document Processor**: Uses backend's document processing pipeline
3. **Backend RAG Tool**: Uses backend's RAG tool for all evaluations  
4. **Backend Session Management**: Creates collections with backend naming (`policy_documents_{session_id}`)
5. **Configuration Inheritance**: Inherits all settings from backend

### Backend Collection Structure
```python
# Always uses backend document processor
doc_processor = BackendDocumentProcessor()
vectorstore = doc_processor.process_uploaded_files(file_objects, session_id)

# Creates collections like: "policy_documents_eval_20241201_143022"
# Same location as backend: ":memory:" or configured URL
```

### No Fallbacks - Backend Required
```python
# FAILS FAST if backend not available
try:
    from tools.rag_tool import RAGTool
    from utils.document_processor import DocumentProcessor
    from config import settings as backend_settings
except ImportError as e:
    print(f"❌ Failed to import backend components: {e}")
    sys.exit(1)  # No fallback - requires backend
```

## 📁 Simplified Architecture

```
evals/
├── golden_dataset_evaluation.py    # 🎯 COMPLETE IMPLEMENTATION (single file!)
├── setup_evaluation.py            # Environment validation and setup
├── pyproject.toml                 # uv dependencies
├── .env                           # API keys (generated by setup)
├── .gitignore                     # Excludes .env and results
├── results/                       # Generated evaluation results
└── README.md                      # This documentation
```

**Everything consolidated into one file!** No more scattered implementation across multiple modules.

## RAGAS TestsetGenerator Implementation

### Exact Implementation (As Specified)
```python
class RAGASTestsetGenerator:
    def __init__(self):
        # Create LangChain wrappers as shown in notebook
        self.generator_llm = LangchainLLMWrapper(
            ChatOpenAI(model="gpt-4o", temperature=0.7)
        )
        self.generator_embeddings = LangchainEmbeddingsWrapper(
            OpenAIEmbeddings(model="text-embedding-3-small")
        )
        
        # Initialize RAGAS TestsetGenerator
        self.generator = TestsetGenerator(
            llm=self.generator_llm,
            embedding_model=self.generator_embeddings
        )
    
    def generate_questions_from_documents(self, documents, testset_size=20):
        # Sample documents as shown in example
        sample_docs = random.sample(documents, min(50, len(documents)))
        
        # Generate testset using RAGAS
        testset = self.generator.generate_with_langchain_docs(
            documents=sample_docs,
            testset_size=testset_size
        )
        
        # Extract questions as shown
        test_questions = list(testset.to_pandas()['user_input'])
        return test_questions
```

## Usage Examples

### All Evaluation Options
```bash
# Complete evaluation with simplified pandas pattern (default: 20 questions)
uv run golden_dataset_evaluation.py

# Custom testset size - uses: test_questions = list(testset.to_pandas()['user_input'])
uv run golden_dataset_evaluation.py --testset-size 50

# With interactive LangSmith setup
uv run golden_dataset_evaluation.py --langsmith-interactive

# Skip LangSmith entirely
uv run golden_dataset_evaluation.py --skip-langsmith
```

### Environment Validation
```bash
# Check setup status
uv run setup_evaluation.py

# This validates:
# - uv installation ✅
# - Dependencies ✅  
# - Backend integration ✅
# - API keys configuration ✅
# - Documents directory ✅
```

### Results Analysis
```bash
# All results automatically saved to results/
ls results/
# golden_dataset_20241201_143022.json       - RAGAS generated questions
# baseline_evaluation_20241201_143022.json  - Raw RAGAS metrics  
# evaluation_summary_20241201_143022.txt    - Human-readable summary

# View latest summary
cat results/evaluation_summary_*.txt
```

## Expected Output

### Task 5 Baseline Results
```
🎯 Final RAGAS Metrics vs Task 5 Expected Baseline:
------------------------------------------------------------
   • context_precision : 0.857 (expected: 0.850) ✅
   • context_recall    : 0.803 (expected: 0.810) ⚠️ 
   • faithfulness     : 0.884 (expected: 0.870) ✅
   • answer_relevancy : 0.925 (expected: 0.920) ✅

🎯 Task 5 Key Finding Confirmed:
   Context recall of 0.803 indicates the naive retriever
   sometimes fails to find all relevant information,
   representing a key area for improvement.
```

### LangSmith Integration Output
```
🔗 STEP 1: SETUP LANGSMITH DATASET
------------------------------------------------------------
🔗 Creating LangSmith client...
📊 Creating dataset: goldendataset-20241201-143022
✅ Dataset URL: https://smith.langchain.com/datasets/abc123
✅ Dataset: goldendataset-20241201-143022
```

## Configuration

### Environment Variables (.env)
```bash
# Required
OPENAI_API_KEY=your_openai_api_key_here

# LangSmith (Optional)
LANGCHAIN_API_KEY=your_langchain_api_key_here
LANGSMITH_PROJECT=policy-analyzer-evals

# Evaluation Settings
TESTSET_SIZE=20
LLM_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
```

### Programmatic Configuration
```python
from config import config

# Inherited from backend
config.llm_model              # gpt-4o-mini
config.embedding_model        # text-embedding-3-small
config.chunk_size            # 1000 (from backend)
config.chunk_overlap         # 200 (from backend)
config.qdrant_location       # ":memory:" (from backend)
config.qdrant_collection_name # "policy_documents" (from backend)

# Evaluation specific
config.testset_size          # 20
config.testset_generator_model  # gpt-4o
```

## Troubleshooting

### Common Issues (All Fail Hard Now)

**1. Missing API Keys (HARD FAILURE)**
```bash
# System fails immediately with clear error:
# ValueError: OPENAI_API_KEY environment variable is required

# Set your API key:
export OPENAI_API_KEY="your_actual_api_key"
# Or edit .env file
```

**2. Backend integration fails (HARD FAILURE)**
```bash
# System exits with ImportError if backend unavailable
# Backend is REQUIRED - setup backend first
cd ../backend && uv sync

# Verify backend works
cd ../backend && uv run python -c "from tools.rag_tool import RAGTool; print('Backend OK')"
```

**3. LangSmith API key missing (HARD FAILURE when enabled)**
```bash
# If you use LangSmith (not --skip-langsmith), system requires:
export LANGSMITH_API_KEY="your_langsmith_key"

# Or skip it entirely:
uv run golden_dataset_evaluation.py --skip-langsmith
```

**4. No documents found (HARD FAILURE)**
```bash
# System fails if no documents in ../documents/
mkdir -p ../documents
cp your_policy_docs.pdf ../documents/
```

## Validation & Testing

### Setup Validation
```bash
# Comprehensive setup check
uv run setup.py

# Component testing
uv run test_evaluation.py
```

### Quick Validation
```bash
# Validate complete setup
uv run setup_evaluation.py

# Test basic functionality (without full evaluation)
uv run golden_dataset_evaluation.py --testset-size 2
```

## Files Generated

After running `uv run golden_dataset_evaluation.py`:
- `golden_dataset_YYYYMMDD_HHMMSS.json` - Simple list of questions: `{"questions": [...], "count": N}`
- `baseline_evaluation_YYYYMMDD_HHMMSS.json` - Raw RAGAS metrics and evaluation data
- `evaluation_summary_YYYYMMDD_HHMMSS.txt` - Human-readable summary with Task 5 analysis

## Implementation Completeness

- ✅ **RAGAS TestsetGenerator**: Uses pandas pattern `list(testset.to_pandas()['user_input'])`
- ✅ **Simplified Question Storage**: No complex knowledge graph - just list of strings
- ✅ **Backend-Only Architecture**: Always uses backend - no fallbacks
- ✅ **Backend Qdrant Integration**: Uses backend Qdrant infrastructure exclusively  
- ✅ **Backend RAG Tool**: Uses backend RAG tool exclusively
- ✅ **Backend Document Processor**: Uses backend document processing exclusively
- ✅ **Hard Failures Only**: No graceful exception handling - fails fast on any error
- ✅ **API Key Validation**: Hard fails immediately if required API keys missing
- ✅ **uv Package Management**: Modern Python package management
- ✅ **Environment Variables**: Proper .env handling with gitignore
- ✅ **LangSmith Integration**: Hard fails if enabled but API key missing
- ✅ **Task 5 Baseline**: Expected metrics and key findings
- ✅ **Single File**: Complete implementation in one file

---

**Status**: ✅ **COMPLETE** - Task 5 implemented in a **single consolidated file** with **backend-only architecture**.

## 🎉 **Key Improvements**

✅ **Single File**: Complete implementation in `golden_dataset_evaluation.py`  
✅ **Backend-Only**: Always uses backend - no fallback code  
✅ **Hard Failures**: No graceful exception handling - fails immediately on any error  
✅ **API Key Validation**: Fails hard if OPENAI_API_KEY or LANGSMITH_API_KEY missing  
✅ **Simplified Questions**: Uses pandas pattern `list(testset.to_pandas()['user_input'])`  
✅ **No Knowledge Graph**: Simple list of strings instead of complex objects  
✅ **Always Consistent**: Uses backend Qdrant, RAG tool, and document processor exclusively  
✅ **RAGAS TestsetGenerator**: Clean pandas extraction pattern  
✅ **One Command**: `uv run golden_dataset_evaluation.py` does everything  

This provides a clean, single-file implementation that **requires and always uses your backend infrastructure** with no fallback complexity.
