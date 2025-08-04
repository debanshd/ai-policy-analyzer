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
    
    def _run(self, query: str, callback_handler=None, **kwargs) -> Dict[str, Any]:
        """
        Execute Data Commons query using data_gemma
        
        Args:
            query: User question that might need statistical data
            callback_handler: Optional callback for streaming updates
            
        Returns:
            DataCommonsResult with real statistical data
        """
        def emit_update(update_type: str, content: str, metadata=None):
            """Helper to emit updates if callback handler is available"""
            if callback_handler:
                callback_handler(update_type, content, metadata)
        
        emit_update("debug", f"🏛️ DataCommons: Processing query: {query[:100]}...")
        print(f"🏛️ DataCommonsTool: Processing query: {query[:100]}...")
        
        if not self.rig_model and not self.rag_model:
            emit_update("debug", "❌ DataCommons tool not properly initialized")
            raise Exception("DataCommons tool not properly initialized")
        
        try:
            emit_update("debug", "📊 Fetching real data from Data Commons...")
            # Use ThreadPoolExecutor to prevent blocking
            with ThreadPoolExecutor() as executor:
                future = executor.submit(self._fetch_real_data, query, emit_update)
                result = future.result(timeout=20)  # 20 second timeout
                emit_update("debug", f"✅ DataCommons: Retrieved {len(result.get('data_points', []))} data points")
                print(f"✅ DataCommonsTool: Retrieved real data")
                return result
                    
        except Exception as e:
            emit_update("debug", f"❌ DataCommons error: {str(e)}")
            print(f"❌ DataCommonsTool: Error fetching data: {e}")
            raise e
    
    def _fetch_real_data(self, query: str, emit_update=None) -> Dict[str, Any]:
        """
        Fetch real statistical data using data_gemma
        
        Args:
            query: User question
            emit_update: Optional callback for streaming updates
            
        Returns:
            Dictionary with real Data Commons results
        """
        if emit_update:
            emit_update("debug", "🔄 Using RIG methodology...")
        print(f"📊 Fetching real data from Data Commons...")
        
        try:
            if self.use_rig and self.rig_model:
                # Use RIG (Retrieval Interleaved Generation) approach
                if emit_update:
                    emit_update("debug", "🔄 Using RIG methodology...")
                print("🔄 Using RIG methodology...")
                response = self.rig_model.query(query)
                
                if emit_update:
                    emit_update("debug", "🔍 Extracting statistical data from response...")
                # Extract statistical data from response
                data_points = self._extract_data_from_response(response, "RIG")
                
            elif self.rag_model:
                # Use RAG (Retrieval Augmented Generation) approach  
                if emit_update:
                    emit_update("debug", "🔍 Using RAG methodology...")
                print("🔍 Using RAG methodology...")
                response = self.rag_model.query(query)
                
                if emit_update:
                    emit_update("debug", "🔍 Extracting statistical data from response...")
                # Extract statistical data from response
                data_points = self._extract_data_from_response(response, "RAG")
                
            else:
                if emit_update:
                    emit_update("debug", "❌ No data_gemma models available")
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
        Extract structured data points from data_gemma response with URLs when available
        
        Args:
            response: Response from data_gemma model
            methodology: "RIG" or "RAG"
            
        Returns:
            List of structured data points with URLs
        """
        data_points = []
        
        try:
            # Convert response to string if needed
            response_text = str(response)
            
            # Extract actual URLs from the response
            dc_urls = self._extract_real_urls_from_response(response_text)
            
            # Look for statistical patterns in the response
            lines = response_text.split('\n')
            
            for i, line in enumerate(lines):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                    
                statistical_keywords = [
                    '%', 'percent', 'million', 'billion', 'thousand', '$', 'tons', 'metric', 
                    'production', 'increased', 'decreased', 'growth', 'decline', 'trend',
                    'tonnes', 'kt', 'mt', 'year', 'annual', 'quarterly', 'monthly',
                    'exports', 'imports', 'revenue', 'value', 'price', 'cost',
                    'australia', 'australian', 'country', 'global', 'world'
                ]
                if any(keyword in line.lower() for keyword in statistical_keywords):
                    # This line likely contains statistical data
                    data_point = {
                        "value": line,
                        "type": "statistic", 
                        "source": f"Data Commons ({methodology})",
                        "confidence": "high"
                    }
                    
                    # Add real URL if available (prefer URLs that appear near this data point)
                    if dc_urls:
                        # Use the first available URL - in future could be more sophisticated
                        data_point["url"] = dc_urls[0]
                        # Remove the URL so next data point gets the next URL
                        dc_urls = dc_urls[1:]
                    
                    data_points.append(data_point)
            
            # If no specific data points found, treat the whole response as one data point
            if not data_points:
                data_point = {
                    "value": response_text[:200] + "..." if len(response_text) > 200 else response_text,
                    "type": "general_data",
                    "source": f"Data Commons ({methodology})",
                    "confidence": "medium"
                }
                
                # Add real URL if available from the response
                remaining_urls = self._extract_real_urls_from_response(response_text)
                if remaining_urls:
                    data_point["url"] = remaining_urls[0]
                # Don't add any URL if none found in the actual response
                    
                data_points.append(data_point)
                
        except Exception as e:
            print(f"⚠️ Error extracting data points: {e}")
            data_points.append({
                "value": f"Data retrieved via {methodology} methodology",
                "type": "metadata",
                "source": f"Data Commons ({methodology})",
                "confidence": "low"
                # No URL added for error case
            })
        
        return data_points[:3]  # Limit to 3 most relevant points
    
    def _extract_real_urls_from_response(self, response_text: str) -> List[str]:
        """
        Extract actual Data Commons URLs from the data_gemma response
        
        Args:
            response_text: Raw response text from data_gemma
            
        Returns:
            List of actual URLs found in the response
        """
        import re
        
        # Look for actual Data Commons URLs in the response
        url_patterns = [
            r'https?://(?:www\.)?datacommons\.org/[^\s\)\]\>]+',
            r'https?://datacommons\.org/[^\s\)\]\>]+',
        ]
        
        urls = []
        for pattern in url_patterns:
            found_urls = re.findall(pattern, response_text)
            urls.extend(found_urls)
        
        # Remove duplicates while preserving order
        unique_urls = []
        for url in urls:
            if url not in unique_urls:
                unique_urls.append(url)
        
        return unique_urls

    async def _arun(self, query: str) -> Dict[str, Any]:
        """Async version - runs sync version in thread pool"""
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            return await loop.run_in_executor(executor, self._run, query) 