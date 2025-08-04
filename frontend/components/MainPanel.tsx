'use client'

import React, { useState, useRef, useEffect } from 'react'
import { AnalysisSession, ChatMessage, UploadedFile, StreamingUpdate } from '@/types'
import FileUpload from './FileUpload'
import ChatInterface from './ChatInterface'

interface MainPanelProps {
  currentSession: AnalysisSession | null
  messages: ChatMessage[]
  onAddFiles: (files: UploadedFile[]) => void
  onAddMessage: (message: Omit<ChatMessage, 'id' | 'timestamp'>) => void
  onUpdateStreamingProgress?: (updates: StreamingUpdate[]) => void
}

export default function MainPanel({
  currentSession,
  messages,
  onAddFiles,
  onAddMessage,
  onUpdateStreamingProgress,
}: MainPanelProps) {
  const [isUploading, setIsUploading] = useState(false)
  const [uploadError, setUploadError] = useState<string | null>(null)



  const handleFileUpload = async (files: File[]) => {
    if (!currentSession) {
      console.error('No current session available for file upload')
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
    if (!content.trim()) return
    
    // Generate a session ID if we don't have one
    const sessionId = currentSession?.id || `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
    
    // Clear any existing streaming updates when starting a new query
    console.log('🧹 Clearing previous analysis updates for new query')
    setCurrentUpdates([])
    onUpdateStreamingProgress?.([])
    completionRef.current = false // Reset completion flag
    
    // Add user message
    onAddMessage({
      type: 'user',
      content: content.trim(),
    })
    
    try {
      // Send query to backend with streaming
      const formData = new FormData()
      formData.append('question', content.trim())
      formData.append('session_id', sessionId)
      
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
      
      console.log('🔄 Starting streaming response processing...')
      
      if (reader) {
        let isStreamComplete = false
        let updateCount = 0
        while (!isStreamComplete) {
          const { done, value } = await reader.read()
          if (done) {
            console.log('✅ Stream completed, processed', updateCount, 'updates')
            isStreamComplete = true
            break
          }
          
          const chunk = decoder.decode(value)
          const lines = chunk.split('\n')
          
          for (const line of lines) {
            if (line.trim() === '') continue // Skip empty lines
            
            if (line.startsWith('data: ')) {
              try {
                const jsonData = line.substring(6)
                const data = JSON.parse(jsonData)
                updateCount++
                
                // Check if this is the final message first
                if (data.type === 'result' && data.metadata?.answer) {
                  console.log('🏁 Final result received, completing stream')
                  // This is the final result - handle completion
                  handleStreamComplete(data)
                  isStreamComplete = true
                } else if (data.type === 'error') {
                  console.log('❌ Error received:', data.content)
                  // Handle error and complete stream
                  onAddMessage({
                    type: 'assistant',
                    content: `Error: ${data.content}`,
                  })
                  onUpdateStreamingProgress?.([])
                  setCurrentUpdates([])
                  isStreamComplete = true
                } else {
                  // Regular streaming update
                  handleStreamingUpdate(data)
                }
              } catch (e) {
                console.error('❌ Error parsing SSE data:', e, 'Raw line:', line)
              }
            }
          }
        }
      } else {
        console.error('❌ No reader available from response')
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

  const [currentUpdates, setCurrentUpdates] = useState<StreamingUpdate[]>([])
  const completionRef = useRef(false)

  const handleStreamingUpdate = (update: any) => {
    // Create the update object
    const newUpdate: StreamingUpdate = {
      type: update.type,
      content: update.content,
      tool: update.tool,
      metadata: update.metadata
    }
    
    // Add to streaming updates - useEffect will handle sending to details panel
    setCurrentUpdates(prev => {
      const newUpdates = [...prev, newUpdate]
      console.log('📤 Adding update to state:', newUpdates.length, '-', update.type)
      return newUpdates
    })
  }

  // Send updates to details panel via useEffect (proper React pattern)
  useEffect(() => {
    console.log('📤 Sending updates to details panel via useEffect:', currentUpdates.length)
    onUpdateStreamingProgress?.(currentUpdates)
  }, [currentUpdates, onUpdateStreamingProgress])

  const handleStreamComplete = (finalUpdate: any) => {
    // Prevent duplicate completions
    if (completionRef.current) {
      return
    }
    
    completionRef.current = true
    
    console.log('🏁 Stream completing with final update:', finalUpdate.type)
    
    // Get content for final message
    const content = finalUpdate.metadata?.answer || finalUpdate.content || 'Analysis complete'
    
    // Add final message with current streaming updates
    onAddMessage({
      type: 'assistant',
      content: content,
      streamingUpdates: [...currentUpdates] // Capture all streaming updates
    })
    
    // Keep updates visible until next prompt - don't clear automatically
    console.log('✅ Analysis complete - keeping updates visible until next prompt')
  }

  if (!currentSession) {
    return (
      <div className="flex-1 flex flex-col">
        <div className="flex-1 flex items-center justify-center">
          <div className="text-center">
            <div className="w-16 h-16 bg-border-light rounded-lg flex items-center justify-center mb-4 mx-auto">
              <svg className="w-8 h-8 text-text-secondary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
              </svg>
            </div>
            <h3 className="text-lg font-medium text-text-primary mb-2">No Active Session</h3>
            <p className="text-text-secondary">Create a new analysis session to get started</p>
          </div>
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