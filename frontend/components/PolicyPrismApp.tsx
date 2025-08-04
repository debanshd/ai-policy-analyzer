'use client'

import React, { useState, useEffect, useCallback } from 'react'
import Sidebar from './Sidebar'
import MainPanel from './MainPanel'
import DetailsPanel from './DetailsPanel'
import SettingsModal from './SettingsModal'
import { AnalysisSession, ChatMessage, UploadedFile, StreamingUpdate } from '@/types'

export default function PolicyPrismApp() {
  const [sessions, setSessions] = useState<AnalysisSession[]>([])
  const [currentSession, setCurrentSession] = useState<AnalysisSession | null>(null)
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [isDetailsPanelOpen, setIsDetailsPanelOpen] = useState(true)
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false)
  const [isSettingsOpen, setIsSettingsOpen] = useState(false)
  const [backendReady, setBackendReady] = useState(false)
  const [setupRequired, setSetupRequired] = useState(false)
  const [currentStreamingUpdates, setCurrentStreamingUpdates] = useState<StreamingUpdate[]>([])
  const [isProcessing, setIsProcessing] = useState(false)

  // Create a new analysis session
  const createNewSession = () => {
    const newSession: AnalysisSession = {
      id: generateSessionId(),
      name: `Analysis ${new Date().toLocaleDateString()}`,
      files: [],
      createdAt: new Date(),
      lastActive: new Date(),
    }
    
    setSessions(prev => [newSession, ...prev])
    setCurrentSession(newSession)
    setMessages([])
    
    return newSession
  }

  // Add files to current session
  const addFilesToSession = (files: UploadedFile[]) => {
    if (!currentSession) return
    
    const updatedSession = {
      ...currentSession,
      files: [...currentSession.files, ...files],
      lastActive: new Date(),
    }
    
    setCurrentSession(updatedSession)
    setSessions(prev => 
      prev.map(s => s.id === currentSession.id ? updatedSession : s)
    )
  }

  // Add message to current session
  const addMessage = (message: Omit<ChatMessage, 'id' | 'timestamp'>) => {
    const newMessage: ChatMessage = {
      ...message,
      id: generateMessageId(),
      timestamp: new Date(),
    }
    
    setMessages(prev => [...prev, newMessage])
    
    // If this is a user message, start processing
    if (message.type === 'user') {
      setIsProcessing(true)
      setCurrentStreamingUpdates([])
    }
    
    // If this is an assistant message with a complete response, stop processing
    if (message.type === 'assistant' && message.content && !message.streamingUpdates) {
      setIsProcessing(false)
      setCurrentStreamingUpdates([])
    }
    
    // Update session last active time
    if (currentSession) {
      const updatedSession = {
        ...currentSession,
        lastActive: new Date(),
      }
      setCurrentSession(updatedSession)
      setSessions(prev => 
        prev.map(s => s.id === currentSession.id ? updatedSession : s)
      )
    }
  }

  // Update streaming updates for the details panel
  const updateStreamingProgress = useCallback((updates: StreamingUpdate[]) => {
    console.log('🎯 PolicyPrismApp received updates:', updates.length, updates.map(u => u.type).join(', '))
    setCurrentStreamingUpdates(updates)
    setIsProcessing(updates.length > 0)
  }, [])



  // Generate unique IDs
  const generateSessionId = () => Math.random().toString(36).substring(2, 15)
  const generateMessageId = () => Math.random().toString(36).substring(2, 15)

  // Check backend status on mount
  useEffect(() => {
    checkBackendStatus()
  }, [])

  // Create initial session immediately on mount
  useEffect(() => {
    if (sessions.length === 0 && !currentSession) {
      createNewSession()
    }
  }, [sessions.length, currentSession])

  // Check backend status and show settings if needed
  const checkBackendStatus = async () => {
    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
      console.log('Checking backend at:', apiUrl)
      const response = await fetch(`${apiUrl}/health`)
      if (response.ok) {
        const data = await response.json()
        console.log('Backend response:', data)
        setBackendReady(data.agent_ready)
        setSetupRequired(!data.keys_configured)
        
        // Auto-open settings if setup is required
        if (!data.keys_configured) {
          setIsSettingsOpen(true)
        }
      }
    } catch (error) {
      console.error('Failed to check backend status:', error)
      setSetupRequired(true)
      setIsSettingsOpen(true)
    }
  }

  // Handle saving API keys
  const handleSaveApiKeys = async (keys: any) => {
    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/settings`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(keys),
      })
      
      if (!response.ok) {
        throw new Error('Failed to save API keys')
      }
      
      const result = await response.json()
      setBackendReady(result.agent_ready)
      setSetupRequired(!result.status.is_valid)
      
      // Show success message
      alert('API keys saved successfully! The backend is now ready.')
    } catch (error) {
      console.error('Failed to save API keys:', error)
      throw error
    }
  }

  // Show setup screen if backend is not ready
  if (!backendReady && setupRequired) {
    return (
      <div className="flex h-screen bg-background">
        <div className="flex-1 flex items-center justify-center p-8">
          <div className="text-center max-w-md">
            <div className="w-20 h-20 bg-accent-blue-light rounded-lg flex items-center justify-center mb-6 mx-auto">
              <svg className="w-10 h-10 text-accent-blue" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
              </svg>
            </div>
            <h1 className="text-2xl font-bold text-text-primary mb-4">Welcome to Policy Prism</h1>
            <p className="text-text-secondary mb-6">
              To get started, please configure your API keys. This enables document analysis, 
              web search, and statistical data integration.
            </p>
            <button
              onClick={() => setIsSettingsOpen(true)}
              className="btn-primary text-lg px-8 py-3"
            >
              Configure API Keys
            </button>
          </div>
        </div>
        
        <SettingsModal
          isOpen={isSettingsOpen}
          onClose={() => setIsSettingsOpen(false)}
          onSave={handleSaveApiKeys}
        />
      </div>
    )
  }

  return (
    <div className="flex h-screen bg-background">
      {/* Left Sidebar */}
      <div className={`transition-all duration-300 ${isSidebarCollapsed ? 'w-16' : 'w-80'}`}>
        <Sidebar
          isCollapsed={isSidebarCollapsed}
          onToggleCollapse={() => setIsSidebarCollapsed(!isSidebarCollapsed)}
          onOpenSettings={() => setIsSettingsOpen(true)}
        />
      </div>

      {/* Main Panel */}
      <div className="flex-1 flex flex-col min-w-0">
        <MainPanel
          currentSession={currentSession}
          messages={messages}
          onAddFiles={addFilesToSession}
          onAddMessage={addMessage}
          onUpdateStreamingProgress={updateStreamingProgress}
        />
      </div>

      {/* Right Details Panel */}
      {isDetailsPanelOpen && (
        <div className="w-80 border-l border-border-light">
          <DetailsPanel
            currentSession={currentSession}
            currentStreamingUpdates={currentStreamingUpdates}
            isProcessing={isProcessing}
            onClose={() => setIsDetailsPanelOpen(false)}
          />
        </div>
      )}

      {/* Toggle Details Panel Button */}
      {!isDetailsPanelOpen && (
        <button
          onClick={() => setIsDetailsPanelOpen(true)}
          className="fixed top-4 right-4 p-2 bg-surface border border-border rounded-lg shadow-sm hover:shadow-md transition-shadow"
          aria-label="Show details panel"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
          </svg>
        </button>
      )}
      
      {/* Settings Modal */}
      <SettingsModal
        isOpen={isSettingsOpen}
        onClose={() => setIsSettingsOpen(false)}
        onSave={handleSaveApiKeys}
      />
    </div>
  )
} 