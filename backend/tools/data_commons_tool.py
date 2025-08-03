"""
Data Commons Tool using data_gemma library for real statistical data retrieval
This replaces the previous simulation with actual Data Commons integration
"""

from typing import Dict, Any, Optional, List
from langchain_core.tools import BaseTool
from pydantic import Field
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor

from config import settings
from models import DataCommonsResult, ToolType

try:
    # Import data_gemma library
    from data_gemma import RIGFlow, RAGFlow, OpenAI, DataCommons
    DATA_GEMMA_AVAILABLE = True
except ImportError:
    DATA_GEMMA_AVAILABLE = False
    print("⚠️ data_gemma library not available. Install with: pip install git+https://github.com/datacommonsorg/llm-tools.git")

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
        if DATA_GEMMA_AVAILABLE:
            self._initialize_data_gemma()
        else:
            print("⚠️ DataCommonsTool: data_gemma not available, falling back to simulation")
    
    def _initialize_data_gemma(self):
        """Initialize data_gemma models"""
        try:
            print("🏗️ Initializing data_gemma components...")
            
            # Get OpenAI API key from environment
            import os
            openai_key = os.getenv("OPENAI_API_KEY")
            if not openai_key:
                raise ValueError("OPENAI_API_KEY not found in environment")
            
            # Get Data Commons API key (optional)
            dc_api_key = os.getenv("DATA_COMMONS_API_KEY")
            if not dc_api_key:
                print("⚠️ DATA_COMMONS_API_KEY not found - data_gemma requires Google Data Commons API access")
                print("📝 To get real data, visit: https://datacommons.org/api")
                print("📝 Set DATA_COMMONS_API_KEY in your .env file")
                raise ValueError("DATA_COMMONS_API_KEY not found in environment")
            
            # Initialize OpenAI LLM for data_gemma
            llm = OpenAI(model=settings.llm_model, api_key=openai_key)
            print("✅ OpenAI LLM initialized")
            
            # Initialize Data Commons data fetcher
            data_fetcher = DataCommons(api_key=dc_api_key)
            print("✅ DataCommons fetcher initialized")
            
            # Initialize RIG flow (Retrieval Interleaved Generation)
            object.__setattr__(self, 'rig_model', RIGFlow(llm=llm, data_fetcher=data_fetcher))
            print("✅ RIG flow initialized")
            
            # Initialize RAG flow (Retrieval Augmented Generation) - needs two LLMs
            object.__setattr__(self, 'rag_model', RAGFlow(llm_question=llm, llm_answer=llm, data_fetcher=data_fetcher))
            print("✅ RAG flow initialized")
            
        except Exception as e:
            print(f"⚠️ Failed to initialize data_gemma: {e}")
            if "DATA_COMMONS_API_KEY" in str(e):
                print("💡 Falling back to simulation mode - no additional setup required")
            else:
                import traceback
                traceback.print_exc()
            object.__setattr__(self, 'rig_model', None)
            object.__setattr__(self, 'rag_model', None)
    
    def _run(self, query: str) -> Dict[str, Any]:
        """
        Execute Data Commons query using data_gemma
        
        Args:
            query: User question that might need statistical data
            
        Returns:
            DataCommonsResult with real statistical data
        """
        print(f"🏛️ DataCommonsTool: Processing query: {query[:100]}...")
        
        if not DATA_GEMMA_AVAILABLE or not self.rig_model:
            print("⚠️ DataCommonsTool: Falling back to simulation mode")
            return self._simulate_data_commons(query)
        
        try:
            # Use ThreadPoolExecutor to prevent blocking
            with ThreadPoolExecutor() as executor:
                future = executor.submit(self._fetch_real_data, query)
                try:
                    result = future.result(timeout=20)  # 20 second timeout
                    print(f"✅ DataCommonsTool: Retrieved real data")
                    return result
                except Exception as e:
                    print(f"⚠️ DataCommonsTool: Timeout or error: {e}")
                    return self._simulate_data_commons(query)
                    
        except Exception as e:
            print(f"❌ DataCommonsTool: Error fetching real data: {e}")
            return self._simulate_data_commons(query)
    
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
            # This is a simplified extraction - data_gemma responses should contain
            # structured statistical data that we can parse
            
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
    
    def _simulate_data_commons(self, query: str) -> Dict[str, Any]:
        """
        Fallback simulation when data_gemma is not available
        """
        print("🎭 Using Data Commons simulation (data_gemma not available)")
        
        # Simple simulation based on query keywords
        simulation_data = []
        
        if any(word in query.lower() for word in ['renewable', 'energy', 'solar', 'wind']):
            simulation_data = [
                {"value": "Renewable energy accounts for 12.2% of total U.S. energy consumption", "type": "statistic", "source": "Data Commons (simulated)", "confidence": "simulated"},
                {"value": "Solar capacity grew by 43% in 2021", "type": "trend", "source": "Data Commons (simulated)", "confidence": "simulated"}
            ]
        elif any(word in query.lower() for word in ['education', 'school', 'student']):
            simulation_data = [
                {"value": "89.8% high school graduation rate in the U.S.", "type": "statistic", "source": "Data Commons (simulated)", "confidence": "simulated"},
                {"value": "Average student-teacher ratio is 16:1", "type": "ratio", "source": "Data Commons (simulated)", "confidence": "simulated"}
            ]
        elif any(word in query.lower() for word in ['healthcare', 'medical', 'hospital']):
            simulation_data = [
                {"value": "17.8% of GDP spent on healthcare", "type": "statistic", "source": "Data Commons (simulated)", "confidence": "simulated"},
                {"value": "2.6 hospital beds per 1,000 people", "type": "ratio", "source": "Data Commons (simulated)", "confidence": "simulated"}
            ]
        else:
            simulation_data = [
                {"value": f"Statistical analysis suggests reviewing data related to: {query[:50]}...", "type": "suggestion", "source": "Data Commons (simulated)", "confidence": "simulated"}
            ]
        
        return {
            "tool_name": "data_commons",
            "tool_type": ToolType.DATA_COMMONS_TOOL,
            "query": query,
            "data_points": simulation_data,
            "source": "Data Commons (simulated - data_gemma not available)",
            "methodology": "simulation",
            "success": True,
            "raw_response": "Simulated response - install data_gemma for real data"
        }

    async def _arun(self, query: str) -> Dict[str, Any]:
        """Async version - runs sync version in thread pool"""
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            return await loop.run_in_executor(executor, self._run, query) 