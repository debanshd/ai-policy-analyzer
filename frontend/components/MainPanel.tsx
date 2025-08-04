'use client'

import React, { useState, useRef } from 'react'
import { AnalysisSession, ChatMessage, UploadedFile } from '@/types'
import FileUpload from './FileUpload'
import ChatInterface from './ChatInterface'

interface MainPanelProps {
  currentSession: AnalysisSession | null
  messages: ChatMessage[]
  onAddFiles: (files: UploadedFile[]) => void
  onAddMessage: (message: Omit<ChatMessage, 'id' | 'timestamp'>) => void
}

export default function MainPanel({
  currentSession,
  messages,
  onAddFiles,
  onAddMessage,
}: MainPanelProps) {
  const [isUploading, setIsUploading] = useState(false)
  const [uploadError, setUploadError] = useState<string | null>(null)

  // Debug logging
  console.log('MainPanel render:', { 
    currentSessionId: currentSession?.id, 
    filesCount: currentSession?.files.length || 0,
    messagesCount: messages.length 
  })

  const handleFileUpload = async (files: File[]) => {
    if (!currentSession) {
      console.log('No current session, but proceeding with mock upload for testing')
      // For testing, just log the files and show a success message
      onAddMessage({
        type: 'system',
        content: `Mock upload: Selected ${files.length} file${files.length !== 1 ? 's' : ''}: ${files.map(f => f.name).join(', ')}. Note: Files not actually uploaded in mock mode.`,
      })
      return
    }
    
    setIsUploading(true)
    setUploadError(null)
    
    try {
      // Upload files to backend
      const formData = new FormData()
      files.forEach(file => formData.append('files', file))
      formData.append('session_id', currentSession.id)
      
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/upload`, {
        method: 'POST',
        body: formData,
      })
      
      if (!response.ok) {
        throw new Error(`Upload failed: ${response.statusText}`)
      }
      
      const results = await response.json()
      
      // Convert to UploadedFile format
      const uploadedFiles: UploadedFile[] = results.map((result: any) => ({
        id: result.file_id,
        filename: result.filename,
        size: result.size,
        uploadedAt: new Date(),
      }))
      
      onAddFiles(uploadedFiles)
      
      console.log('Files uploaded successfully:', uploadedFiles)
      
      // Add system message about successful upload
      onAddMessage({
        type: 'system',
        content: `Successfully uploaded ${uploadedFiles.length} file${uploadedFiles.length !== 1 ? 's' : ''}: ${uploadedFiles.map(f => f.filename).join(', ')}`,
      })
      
    } catch (error) {
      console.error('Upload error:', error)
      setUploadError(error instanceof Error ? error.message : 'Upload failed')
    } finally {
      setIsUploading(false)
    }
  }

  const handleSendMessage = async (content: string) => {
    if (!currentSession || !content.trim()) return
    
    // Add user message
    onAddMessage({
      type: 'user',
      content: content.trim(),
    })
    
    try {
      // Send query to backend with streaming
      const formData = new FormData()
      formData.append('question', content.trim())
      formData.append('session_id', currentSession.id)
      
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/query`, {
        method: 'POST',
        body: formData,
      })
      
      if (!response.ok) {
        throw new Error(`Query failed: ${response.statusText}`)
      }
      
      // Handle streaming response
      const reader = response.body?.getReader()
      const decoder = new TextDecoder()
      
      if (reader) {
        let isStreamComplete = false
        while (!isStreamComplete) {
          const { done, value } = await reader.read()
          if (done) {
            isStreamComplete = true
            break
          }
          
          const chunk = decoder.decode(value)
          const lines = chunk.split('\n')
          
          for (const line of lines) {
            if (line.startsWith('data: ')) {
              try {
                const data = JSON.parse(line.substring(6))
                // Handle streaming update
                handleStreamingUpdate(data)
                
                // Check if this is the final message
                if (data.type === 'result' || data.done === true) {
                  isStreamComplete = true
                }
              } catch (e) {
                console.error('Error parsing SSE data:', e)
              }
            }
          }
        }
      }
      
    } catch (error) {
      console.error('Query error:', error)
      // Add error message
      onAddMessage({
        type: 'assistant',
        content: `Error: ${error instanceof Error ? error.message : 'Query failed'}`,
      })
      throw error // Re-throw to let ChatInterface handle the error state
    }
  }

  const handleStreamingUpdate = (update: any) => {
    // Update the specific message with streaming data
    if (update.type === 'result' && update.metadata) {
      // Final result with complete answer
      onAddMessage({
        type: 'assistant',
        content: update.metadata.answer || update.content,
      })
    } else if (update.type === 'thought' || update.type === 'tool') {
      // Show intermediate updates as thoughts
      console.log(`[${update.type}] ${update.content}`)
      // You could update the UI here to show these intermediate steps
      // For now, we'll just log them
    } else if (update.type === 'error') {
      // Handle errors
      onAddMessage({
        type: 'assistant',
        content: `Error: ${update.content}`,
      })
    }
  }

  if (!currentSession) {
    // Temporarily create a mock session for testing
    const mockSession: AnalysisSession = {
      id: 'mock-session-' + Date.now(),
      name: 'Test Session',
      files: [],
      createdAt: new Date(),
      lastActive: new Date(),
    }
    console.log('Using mock session for testing')
    
    return (
      <div className="flex-1 flex flex-col">
        {/* File Upload Area - Always show since no files in mock session */}
        <div className="p-6 border-b border-border-light">
          <FileUpload
            onFilesSelected={handleFileUpload}
            isUploading={isUploading}
            error={uploadError}
          />
        </div>

        {/* Chat Interface */}
        <div className="flex-1 flex flex-col min-h-0">
          <ChatInterface
            messages={messages}
            onSendMessage={handleSendMessage}
            disabled={false} // Enable chat for testing - you can upload files first
          />
        </div>
      </div>
    )
  }

  return (
    <div className="flex-1 flex flex-col">
      {/* File Upload Area */}
      {currentSession.files.length === 0 ? (
        <div className="p-6 border-b border-border-light">
          <FileUpload
            onFilesSelected={handleFileUpload}
            isUploading={isUploading}
            error={uploadError}
          />
        </div>
      ) : (
        // Uploaded files summary
        <div className="p-4 border-b border-border-light bg-surface">
          <div className="flex items-center gap-2 text-sm text-text-secondary">
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
            <span>
              {currentSession.files.length} file{currentSession.files.length !== 1 ? 's' : ''} uploaded • Ready for analysis
            </span>
          </div>
        </div>
      )}

      {/* Chat Interface */}
      <div className="flex-1 flex flex-col min-h-0">
        <ChatInterface
          messages={messages}
          onSendMessage={handleSendMessage}
          disabled={false} // Temporarily always enabled for testing
        />
      </div>
    </div>
  )
} 