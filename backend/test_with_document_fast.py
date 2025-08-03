#!/usr/bin/env python3
"""
Fast test script for Multi-Source Analysis Agent - skips external APIs for speed

This is a simplified version that only tests document upload and basic RAG,
without the Tavily web search and Data Commons calls.
"""

import argparse
import asyncio
import json
from pathlib import Path
import aiohttp
import aiofiles

BASE_URL = "http://localhost:8000"

async def upload_document(file_path: str) -> dict:
    """Upload a document to the backend"""
    print(f"📄 Uploading document: {file_path}")
    
    async with aiohttp.ClientSession() as session:
        data = aiohttp.FormData()
        
        # Add file
        async with aiofiles.open(file_path, 'rb') as f:
            file_content = await f.read()
            
            # Determine content type
            if file_path.endswith('.pdf'):
                content_type = 'application/pdf'
            elif file_path.endswith(('.png', '.jpg', '.jpeg')):
                content_type = 'image/jpeg'
            elif file_path.endswith('.txt'):
                content_type = 'text/plain'
            elif file_path.endswith('.md'):
                content_type = 'text/markdown'
            else:
                content_type = 'application/octet-stream'
                
            data.add_field('files', 
                          file_content, 
                          filename=Path(file_path).name,
                          content_type=content_type)
        
        async with session.post(f"{BASE_URL}/upload", data=data) as response:
            if response.status == 200:
                result = await response.json()
                print(f"✅ Upload successful!")
                for file_response in result:
                    print(f"   📁 {file_response['filename']} ({file_response['size']} bytes)")
                    session_id = file_response['message'].split('Session: ')[1].rstrip(')')
                return {"success": True, "session_id": session_id, "files": result}
            else:
                error = await response.text()
                print(f"❌ Upload failed: {error}")
                return {"success": False, "error": error}

async def test_basic_query(prompt: str, session_id: str) -> None:
    """Test basic query without full streaming"""
    print(f"\n🤖 Testing basic query: '{prompt}'")
    print("⚡ Running fast test (RAG only)...\n")
    
    async with aiohttp.ClientSession() as session:
        data = aiohttp.FormData()
        data.add_field('question', prompt)
        data.add_field('session_id', session_id)
        
        # Set a shorter timeout for the test
        timeout = aiohttp.ClientTimeout(total=60)  # 60 second timeout
        
        async with session.post(f"{BASE_URL}/query", data=data, timeout=timeout) as response:
            if response.status == 200:
                print("📡 Receiving response...")
                
                # Process streaming response
                buffer = ""
                async for chunk in response.content.iter_chunked(1024):
                    chunk_text = chunk.decode('utf-8')
                    buffer += chunk_text
                    
                    # Process complete lines
                    lines = buffer.split('\n')
                    buffer = lines[-1]  # Keep incomplete line in buffer
                    
                    for line in lines[:-1]:
                        if line.startswith('data: '):
                            json_data = line[6:]  # Remove 'data: ' prefix
                            try:
                                update = json.loads(json_data)
                                format_update(update)
                                
                                # Stop after we get the final result to speed up testing
                                if update.get('type') == 'result' and update.get('metadata'):
                                    print(f"\n⚡ Fast test completed! (Full analysis may continue in background)")
                                    return
                                    
                            except json.JSONDecodeError:
                                continue
            else:
                error = await response.text()
                print(f"❌ Query failed: {error}")

def format_update(update: dict):
    """Format and display updates"""
    update_type = update.get('type', 'unknown')
    content = update.get('content', '')
    tool = update.get('tool')
    
    if update_type == 'thought':
        print(f"💭 {content}")
    elif update_type == 'tool':
        tool_emoji = {'RAG_Tool': '📚', 'DataCommonsTool': '📊', 'TavilyTool': '🌐'}
        emoji = tool_emoji.get(tool, '🔧')
        print(f"{emoji} [{tool}] {content}")
    elif update_type == 'result':
        if update.get('metadata'):
            metadata = update['metadata']
            print(f"\n🎯 RESULT:")
            print("=" * 40)
            print(metadata['answer'][:300] + "..." if len(metadata['answer']) > 300 else metadata['answer'])
            print("=" * 40)
            print(f"⏱️  Time: {metadata['processing_time']:.1f}s")
        else:
            print(f"✅ {content}")
    elif update_type == 'error':
        print(f"❌ Error: {content}")

async def main():
    parser = argparse.ArgumentParser(description="Fast test for Multi-Source Analysis Agent")
    parser.add_argument("--file", "-f", required=True, help="Path to document file")
    parser.add_argument("--prompt", "-p", required=True, help="Question/prompt to ask")
    
    args = parser.parse_args()
    
    if not Path(args.file).exists():
        print(f"❌ File not found: {args.file}")
        return
    
    print("⚡ Fast Test Mode - Multi-Source Analysis Agent")
    print("=" * 50)
    
    # Upload document
    upload_result = await upload_document(args.file)
    if not upload_result["success"]:
        return
    
    session_id = upload_result["session_id"]
    print(f"🔑 Session ID: {session_id}")
    
    # Wait a moment for processing
    await asyncio.sleep(1)
    
    # Test query
    await test_basic_query(args.prompt, session_id)
    
    print("\n⚡ Fast test completed!")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted")
    except Exception as e:
        print(f"\n❌ Test failed: {e}") 