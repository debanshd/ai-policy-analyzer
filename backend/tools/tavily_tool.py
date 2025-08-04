from typing import Dict, Any, List, Optional
from langchain_core.tools import BaseTool
from tavily import TavilyClient
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import Field

from config import settings
from models import TavilyResult, ToolType

class TavilyTool(BaseTool):
    """
    Tavily search tool for web searches to provide definitions and broader context
    """
    
    name: str = "tavily_tool"
    description: str = "Search the web for definitions, news, and broader context on topics"
    
    # Declare fields
    client: Optional[TavilyClient] = Field(default=None, exclude=True)
    llm: Optional[ChatOpenAI] = Field(default=None, exclude=True)
    summary_prompt: Optional[ChatPromptTemplate] = Field(default=None, exclude=True)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, 'client', TavilyClient(api_key=settings.tavily_api_key))
        object.__setattr__(self, 'llm', ChatOpenAI(
            model=settings.llm_model, 
            temperature=0.1,
            timeout=30,  # 30 second timeout
            max_retries=2
        ))
        object.__setattr__(self, 'summary_prompt', self._create_summary_prompt())
        
    def _create_summary_prompt(self) -> ChatPromptTemplate:
        """Create prompt for summarizing web search results"""
        SUMMARY_TEMPLATE = """You are a research assistant. Based on the web search results below, provide a concise and informative summary that would be useful for a policy analyst or researcher.

Focus on:
- Key definitions and concepts
- Recent developments or news
- Relevant statistics or data points
- Important context or background information

Search Query: {query}

Search Results:
{search_results}

Provide a clear, factual summary:
"""
        return ChatPromptTemplate.from_template(SUMMARY_TEMPLATE)
    
    def _run(self, query: str, max_results: int = 5, **kwargs) -> Dict[str, Any]:
        """Execute web search using Tavily"""
        try:
            print(f"🌐 Tavily searching for: {query[:50]}...")  # Debug output
            
            # Perform the search with timeout handling
            import asyncio
            from concurrent.futures import ThreadPoolExecutor, TimeoutError
            
            def search_with_timeout():
                return self.client.search(
                    query=query,
                    search_depth="basic",
                    max_results=max_results,
                    include_answer=False,
                    include_raw_content=False
                )
            
            # Run with timeout
            with ThreadPoolExecutor() as executor:
                future = executor.submit(search_with_timeout)
                try:
                    search_response = future.result(timeout=15)  # 15 second timeout
                except TimeoutError:
                    print(f"⚠️ Tavily search timed out for: {query}")
                    return {
                        "search_results": [],
                        "summary": f"Search timed out for: {query}",
                        "urls": [],
                        "tool_used": ToolType.TAVILY_TOOL
                    }
            
            # Extract relevant information
            search_results = search_response.get("results", [])
            
            if not search_results:
                return {
                    "search_results": [],
                    "summary": f"No web search results found for: {query}",
                    "urls": [],
                    "tool_used": ToolType.TAVILY_TOOL
                }
            
            # Format search results for summary
            formatted_results = []
            urls = []
            
            for i, result in enumerate(search_results, 1):
                title = result.get("title", "No title")
                content = result.get("content", "No content")
                url = result.get("url", "")
                
                formatted_results.append(f"{i}. {title}\n{content}\nSource: {url}\n")
                urls.append(url)
            
            # Generate summary using LLM
            summary_input = {
                "query": query,
                "search_results": "\n".join(formatted_results)
            }
            
            summary = self.llm.invoke(
                self.summary_prompt.format(**summary_input)
            ).content
            
            return {
                "search_results": search_results,
                "summary": summary,
                "urls": urls,
                "tool_used": ToolType.TAVILY_TOOL
            }
            
        except Exception as e:
            return {
                "search_results": [],
                "summary": f"Error performing web search: {str(e)}",
                "urls": [],
                "error": str(e),
                "tool_used": ToolType.TAVILY_TOOL
            }
    
    async def _arun(self, query: str, max_results: int = 5, **kwargs) -> Dict[str, Any]:
        """Async version of the tool"""
        return self._run(query, max_results, **kwargs)
    
    def search_with_entities(self, entities: List[str], context: str = "") -> Dict[str, Any]:
        """
        Search for multiple entities with additional context
        
        Args:
            entities: List of key terms or entities to search for
            context: Additional context to help with the search
            
        Returns:
            Combined search results for all entities
        """
        try:
            all_results = []
            all_urls = []
            summaries = []
            
            for entity in entities[:2]:  # Reduce to 2 for speed
                # Create a contextualized query
                if context:
                    query = f"{entity} {context}"
                else:
                    query = entity
                
                result = self._run(query, max_results=2)  # Reduce to 2 results for speed
                
                if not result.get("error"):
                    all_results.extend(result.get("search_results", []))
                    all_urls.extend(result.get("urls", []))
                    summaries.append(f"**{entity}**: {result.get('summary', '')}")
            
            combined_summary = "\n\n".join(summaries) if summaries else "No search results found."
            
            return {
                "search_results": all_results,
                "summary": combined_summary,
                "urls": list(set(all_urls)),  # Remove duplicates
                "entities_searched": entities,
                "tool_used": ToolType.TAVILY_TOOL
            }
            
        except Exception as e:
            return {
                "search_results": [],
                "summary": f"Error searching for entities: {str(e)}",
                "urls": [],
                "error": str(e),
                "tool_used": ToolType.TAVILY_TOOL
            } 