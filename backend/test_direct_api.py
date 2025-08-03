#!/usr/bin/env python3
"""
Direct API test to isolate the streaming issue
"""

import asyncio
import aiohttp
import json
import time
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

async def test_basic_endpoints():
    """Test basic API endpoints"""
    print("🔍 Testing basic API endpoints...")
    
    try:
        async with aiohttp.ClientSession() as session:
            # Test health
            print("1. Testing health endpoint...")
            async with session.get("http://localhost:8000/health") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"   ✅ Health check: {data['status']}")
                else:
                    print(f"   ❌ Health check failed: {response.status}")
                    return False
            
            # Test sessions endpoint
            print("2. Testing sessions endpoint...")
            async with session.get("http://localhost:8000/sessions") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"   ✅ Sessions: {data['total']} active")
                else:
                    print(f"   ❌ Sessions failed: {response.status}")
                    return False
                    
            return True
            
    except Exception as e:
        print(f"❌ Basic API test failed: {e}")
        return False

async def test_upload():
    """Test file upload"""
    print("📄 Testing file upload...")
    
    try:
        # Create a simple test file content
        test_content = b"Test policy document with recommendations for carbon pricing and renewable energy."
        
        async with aiohttp.ClientSession() as session:
            data = aiohttp.FormData()
            data.add_field('files', test_content, filename='test.txt', content_type='text/plain')
            
            start_time = time.time()
            async with session.post("http://localhost:8000/upload", data=data) as response:
                upload_time = time.time() - start_time
                
                if response.status == 200:
                    result = await response.json()
                    print(f"   ✅ Upload completed in {upload_time:.2f}s")
                    
                    # Extract session ID
                    session_id = result[0]['message'].split('Session: ')[1].rstrip(')')
                    print(f"   📁 Session ID: {session_id}")
                    return session_id
                else:
                    error = await response.text()
                    print(f"   ❌ Upload failed: {error}")
                    return None
                    
    except Exception as e:
        print(f"❌ Upload test failed: {e}")
        return None

async def test_simple_query(session_id: str):
    """Test a simple query with timeout"""
    print("🤖 Testing simple query...")
    
    try:
        async with aiohttp.ClientSession() as session:
            data = aiohttp.FormData()
            data.add_field('question', 'What are the main points?')
            data.add_field('session_id', session_id)
            
            # Set a timeout
            timeout = aiohttp.ClientTimeout(total=30)  # 30 second timeout
            
            print("   Sending query...")
            start_time = time.time()
            
            async with session.post("http://localhost:8000/query", data=data, timeout=timeout) as response:
                if response.status != 200:
                    error = await response.text()
                    print(f"   ❌ Query failed: {error}")
                    return False
                
                print("   📡 Reading streaming response...")
                response_count = 0
                
                async for line in response.content:
                    line_text = line.decode('utf-8').strip()
                    
                    if line_text.startswith('data: '):
                        response_count += 1
                        elapsed = time.time() - start_time
                        
                        try:
                            json_data = line_text[6:]  # Remove 'data: ' prefix
                            update = json.loads(json_data)
                            update_type = update.get('type', 'unknown')
                            content = update.get('content', '')
                            
                            print(f"   [{elapsed:.1f}s] {update_type}: {content[:50]}...")
                            
                            # Stop after getting some responses to avoid hanging
                            if response_count >= 10 or update_type == 'result':
                                print(f"   ✅ Received {response_count} updates in {elapsed:.2f}s")
                                return True
                                
                        except json.JSONDecodeError:
                            print(f"   📝 Raw data: {line_text[:50]}...")
                            
                        # Safety timeout
                        if elapsed > 25:  # Stop after 25 seconds
                            print(f"   ⚠️ Stopping after {elapsed:.1f}s (safety timeout)")
                            return True
                
                total_time = time.time() - start_time
                print(f"   ✅ Query completed in {total_time:.2f}s with {response_count} updates")
                return True
                
    except asyncio.TimeoutError:
        print(f"   ⚠️ Query timed out after 30s")
        return False
    except Exception as e:
        print(f"❌ Query test failed: {e}")
        return False

async def main():
    print("🚀 Direct API Test")
    print("=" * 40)
    
    # Test basic endpoints
    if not await test_basic_endpoints():
        print("❌ Basic endpoints failed")
        return
    
    # Test upload
    session_id = await test_upload()
    if not session_id:
        print("❌ Upload failed")
        return
    
    # Test query
    query_ok = await test_simple_query(session_id)
    
    if query_ok:
        print("\n✅ Direct API test completed!")
        print("The streaming endpoint is working.")
    else:
        print("\n❌ Query test failed")
        print("The issue is in the query processing.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted")
    except Exception as e:
        print(f"\n❌ Test failed: {e}") 