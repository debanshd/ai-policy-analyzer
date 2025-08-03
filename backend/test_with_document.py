#!/usr/bin/env python3
"""
Test script for Multi-Source Analysis Agent with actual document upload and query

This script demonstrates how to:
1. Upload a document to the backend
2. Query the document with a prompt
3. Stream the agent's response

Usage:
    python test_with_document.py --file path/to/document.pdf --prompt "Your question here"
    python test_with_document.py --help
"""

import argparse
import asyncio
import json
import time
from pathlib import Path
import aiohttp
import aiofiles

BASE_URL = "http://localhost:8000"

async def upload_document(file_path: str, session_id: str = None) -> dict:
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
        
        # Add session_id if provided
        if session_id:
            data.add_field('session_id', session_id)
        
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

async def query_document(prompt: str, session_id: str) -> None:
    """Query the uploaded document with streaming response"""
    print(f"\n🤖 Querying with prompt: '{prompt}'")
    print("📡 Streaming agent response...\n")
    
    async with aiohttp.ClientSession() as session:
        data = aiohttp.FormData()
        data.add_field('question', prompt)
        data.add_field('session_id', session_id)
        
        async with session.post(f"{BASE_URL}/query", data=data) as response:
            if response.status == 200:
                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    if line.startswith('data: '):
                        json_data = line[6:]  # Remove 'data: ' prefix
                        try:
                            update = json.loads(json_data)
                            format_streaming_update(update)
                        except json.JSONDecodeError:
                            print(f"Raw: {line}")
            else:
                error = await response.text()
                print(f"❌ Query failed: {error}")

def format_streaming_update(update: dict):
    """Format and display streaming updates"""
    update_type = update.get('type', 'unknown')
    content = update.get('content', '')
    tool = update.get('tool')
    
    if update_type == 'thought':
        print(f"💭 {content}")
    elif update_type == 'tool':
        tool_emoji = {
            'RAG_Tool': '📚',
            'DataCommonsTool': '📊', 
            'TavilyTool': '🌐'
        }
        emoji = tool_emoji.get(tool, '🔧')
        print(f"{emoji} [{tool}] {content}")
    elif update_type == 'result':
        if update.get('metadata'):
            # Final result with metadata
            metadata = update['metadata']
            print(f"\n🎯 FINAL ANSWER:")
            print("=" * 60)
            print(metadata['answer'])
            print("=" * 60)
            print(f"📋 Sources: {', '.join(metadata['sources'])}")
            print(f"🛠️  Tools used: {', '.join(metadata['tools_used'])}")
            print(f"⏱️  Processing time: {metadata['processing_time']:.2f}s")
        else:
            print(f"✅ {content}")
    elif update_type == 'error':
        print(f"❌ Error: {content}")
    else:
        print(f"📝 {content}")

async def test_health_check():
    """Test if the backend is running"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{BASE_URL}/health") as response:
                if response.status == 200:
                    result = await response.json()
                    print(f"✅ Backend is healthy: {result['status']}")
                    return True
                else:
                    print(f"❌ Backend health check failed: {response.status}")
                    return False
    except Exception as e:
        print(f"❌ Cannot connect to backend: {e}")
        print(f"💡 Make sure the backend is running on {BASE_URL}")
        return False

async def main():
    parser = argparse.ArgumentParser(description="Test Multi-Source Analysis Agent with document")
    parser.add_argument("--file", "-f", required=True, help="Path to document file (PDF or image)")
    parser.add_argument("--prompt", "-p", required=True, help="Question/prompt to ask about the document")
    parser.add_argument("--session", "-s", help="Existing session ID (optional)")
    
    args = parser.parse_args()
    
    # Validate file exists
    if not Path(args.file).exists():
        print(f"❌ File not found: {args.file}")
        return
    
    print("🚀 Testing Multi-Source Analysis Agent")
    print("=" * 50)
    
    # Check if backend is running
    if not await test_health_check():
        return
    
    # Upload document
    upload_result = await upload_document(args.file, args.session)
    if not upload_result["success"]:
        return
    
    session_id = upload_result["session_id"]
    print(f"🔑 Session ID: {session_id}")
    
    # Small delay to ensure processing is complete
    await asyncio.sleep(1)
    
    # Query document
    await query_document(args.prompt, session_id)
    
    print("\n✅ Test completed!")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc() 