"""
Data Commons Tool using data_gemma library for real statistical data retrieval
"""

from typing import Dict, Any, Optional, List
from langchain_core.tools import BaseTool
from pydantic import Field
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor

from config import settings
from models import DataCommonsResult, ToolType

# Import data_gemma library
from data_gemma import RIGFlow, RAGFlow, OpenAI, DataCommons

class DataCommonsTool(BaseTool):
    """
    Real Data Commons tool using data_gemma library for statistical data retrieval.
    Uses both RIG (Retrieval Interleaved Generation) and RAG methodologies.
    """
    
    name: str = "data_commons_tool"
    description: str = "Fetch real statistical data from Data Commons using data_gemma"
    
    # Declare data_gemma components as Pydantic fields
    rig_model: Optional[Any] = Field(default=None, exclude=True)
    rag_model: Optional[Any] = Field(default=None, exclude=True)
    use_rig: bool = Field(default=True, exclude=True)  # Default to RIG methodology
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._initialize_data_gemma()
    
    def _initialize_data_gemma(self):
        """Initialize data_gemma models"""
        try:
            print("🏗️ Initializing data_gemma components...")
            
            # Get OpenAI API key from environment
            import os
            openai_key = os.getenv("OPENAI_API_KEY")
            if not openai_key:
                raise ValueError("OPENAI_API_KEY not found in environment")
            
            # Get Data Commons API key (optional for basic functionality)
            dc_api_key = os.getenv("DATA_COMMONS_API_KEY", "")
            
            # Initialize OpenAI LLM for data_gemma
            llm = OpenAI(model=settings.llm_model, api_key=openai_key)
            print("✅ OpenAI LLM initialized")
            
            # Initialize Data Commons data fetcher
            data_fetcher = DataCommons(api_key=dc_api_key) if dc_api_key else DataCommons()
            print("✅ DataCommons fetcher initialized")
            
            # Initialize RIG flow (Retrieval Interleaved Generation)
            object.__setattr__(self, 'rig_model', RIGFlow(llm=llm, data_fetcher=data_fetcher))
            print("✅ RIG flow initialized")
            
            # Initialize RAG flow (Retrieval Augmented Generation) - needs two LLMs
            object.__setattr__(self, 'rag_model', RAGFlow(llm_question=llm, llm_answer=llm, data_fetcher=data_fetcher))
            print("✅ RAG flow initialized")
            
        except Exception as e:
            print(f"❌ Failed to initialize data_gemma: {e}")
            object.__setattr__(self, 'rig_model', None)
            object.__setattr__(self, 'rag_model', None)
            raise e
    
    def _run(self, query: str) -> Dict[str, Any]:
        """
        Execute Data Commons query using data_gemma
        
        Args:
            query: User question that might need statistical data
            
        Returns:
            DataCommonsResult with real statistical data
        """
        print(f"🏛️ DataCommonsTool: Processing query: {query[:100]}...")
        
        if not self.rig_model and not self.rag_model:
            raise Exception("DataCommons tool not properly initialized")
        
        try:
            # Use ThreadPoolExecutor to prevent blocking
            with ThreadPoolExecutor() as executor:
                future = executor.submit(self._fetch_real_data, query)
                result = future.result(timeout=20)  # 20 second timeout
                print(f"✅ DataCommonsTool: Retrieved real data")
                return result
                    
        except Exception as e:
            print(f"❌ DataCommonsTool: Error fetching data: {e}")
            raise e
    
    def _fetch_real_data(self, query: str) -> Dict[str, Any]:
        """
        Fetch real statistical data using data_gemma
        
        Args:
            query: User question
            
        Returns:
            Dictionary with real Data Commons results
        """
        print(f"📊 Fetching real data from Data Commons...")
        
        try:
            if self.use_rig and self.rig_model:
                # Use RIG (Retrieval Interleaved Generation) approach
                print("🔄 Using RIG methodology...")
                response = self.rig_model.query(query)
                
                # Extract statistical data from response
                data_points = self._extract_data_from_response(response, "RIG")
                
            elif self.rag_model:
                # Use RAG (Retrieval Augmented Generation) approach  
                print("🔍 Using RAG methodology...")
                response = self.rag_model.query(query)
                
                # Extract statistical data from response
                data_points = self._extract_data_from_response(response, "RAG")
                
            else:
                raise ValueError("No data_gemma models available")
            
            return {
                "tool_name": "data_commons",
                "tool_type": ToolType.DATA_COMMONS_TOOL,
                "query": query,
                "data_points": data_points,
                "source": "Data Commons (data_gemma)",
                "methodology": "RIG" if self.use_rig else "RAG",
                "success": True,
                "raw_response": str(response)[:500] + "..." if len(str(response)) > 500 else str(response)
            }
            
        except Exception as e:
            print(f"❌ Error in _fetch_real_data: {e}")
            raise e
    
    def _extract_data_from_response(self, response: Any, methodology: str) -> List[Dict[str, Any]]:
        """
        Extract structured data points from data_gemma response
        
        Args:
            response: Response from data_gemma model
            methodology: "RIG" or "RAG"
            
        Returns:
            List of structured data points
        """
        data_points = []
        
        try:
            # Convert response to string if needed
            response_text = str(response)
            
            # Look for statistical patterns in the response
            lines = response_text.split('\n')
            for line in lines:
                line = line.strip()
                if any(keyword in line.lower() for keyword in ['%', 'percent', 'million', 'billion', 'thousand', '$']):
                    # This line likely contains statistical data
                    data_points.append({
                        "value": line,
                        "type": "statistic", 
                        "source": f"Data Commons ({methodology})",
                        "confidence": "high"
                    })
            
            # If no specific data points found, treat the whole response as one data point
            if not data_points:
                data_points.append({
                    "value": response_text[:200] + "..." if len(response_text) > 200 else response_text,
                    "type": "general_data",
                    "source": f"Data Commons ({methodology})",
                    "confidence": "medium"
                })
                
        except Exception as e:
            print(f"⚠️ Error extracting data points: {e}")
            data_points.append({
                "value": f"Data retrieved via {methodology} methodology",
                "type": "metadata",
                "source": f"Data Commons ({methodology})",
                "confidence": "low"
            })
        
        return data_points[:3]  # Limit to 3 most relevant points

    async def _arun(self, query: str) -> Dict[str, Any]:
        """Async version - runs sync version in thread pool"""
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            return await loop.run_in_executor(executor, self._run, query) 