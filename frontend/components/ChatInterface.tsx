'use client'

import React, { useState, useRef, useEffect } from 'react'
import { ChatMessage, StreamingUpdate } from '@/types'

interface ChatInterfaceProps {
  messages: ChatMessage[]
  onSendMessage: (content: string) => Promise<void>
  disabled?: boolean
}

export default function ChatInterface({ messages, onSendMessage, disabled = false }: ChatInterfaceProps) {
  const [inputValue, setInputValue] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  // Debug logging
  console.log('ChatInterface render:', { disabled, isLoading, messagesLength: messages.length })

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!inputValue.trim() || disabled || isLoading) return

    setIsLoading(true)
    const message = inputValue.trim()
    setInputValue('')
    
    try {
      await onSendMessage(message)
    } finally {
      setIsLoading(false)
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit(e)
    }
  }

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`
    }
  }, [inputValue])

  if (messages.length === 0) {
    return (
      <div className="flex-1 flex flex-col">
        {/* Empty State */}
        <div className="flex-1 flex items-center justify-center p-8">
          <div className="text-center max-w-md">
            <div className="w-16 h-16 bg-accent-blue-light rounded-lg flex items-center justify-center mb-4 mx-auto">
              <svg className="w-8 h-8 text-accent-blue" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
              </svg>
            </div>
            <h3 className="text-lg font-semibold text-text-primary mb-2">
              {disabled ? 'Upload documents to start' : 'Ready for analysis'}
            </h3>
            <p className="text-text-secondary">
              {disabled 
                ? 'Upload your PDF or image files first, then ask questions about their content'
                : 'Ask me anything about your uploaded documents. I can analyze, summarize, and provide insights.'
              }
            </p>
          </div>
        </div>

        {/* Input Form */}
        <div className="border-t border-border-light p-4">
          <form onSubmit={handleSubmit} className="max-w-4xl mx-auto">
            <div className="flex gap-3">
              <div className="flex-1 relative">
                <textarea
                  ref={textareaRef}
                  value={inputValue}
                  onChange={(e) => setInputValue(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder={disabled ? "Upload documents first..." : "Ask a question about your documents..."}
                  disabled={disabled || isLoading}
                  className="w-full resize-none rounded-lg border border-border px-4 py-3 pr-12 focus:border-accent-blue focus:outline-none focus:ring-2 focus:ring-accent-blue focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed"
                  rows={1}
                  style={{ maxHeight: '120px' }}
                />
              </div>
              <button
                type="submit"
                disabled={!inputValue.trim() || disabled || isLoading}
                className="btn-primary px-6 py-3 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isLoading ? (
                  <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                ) : (
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
                  </svg>
                )}
              </button>
            </div>
          </form>
        </div>
      </div>
    )
  }

  return (
    <div className="flex-1 flex flex-col">
      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-6">
        {messages.map((message) => (
          <MessageComponent key={message.id} message={message} />
        ))}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Form */}
      <div className="border-t border-border-light p-4">
        <form onSubmit={handleSubmit} className="max-w-4xl mx-auto">
          <div className="flex gap-3">
            <div className="flex-1 relative">
              <textarea
                ref={textareaRef}
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Ask a follow-up question..."
                disabled={disabled || isLoading}
                className="w-full resize-none rounded-lg border border-border px-4 py-3 pr-12 focus:border-accent-blue focus:outline-none focus:ring-2 focus:ring-accent-blue focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed"
                rows={1}
                style={{ maxHeight: '120px' }}
              />
            </div>
            <button
              type="submit"
              disabled={!inputValue.trim() || disabled || isLoading}
              className="btn-primary px-6 py-3 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isLoading ? (
                <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
              ) : (
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
                </svg>
              )}
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}

// Message Component
function MessageComponent({ message }: { message: ChatMessage }) {
  if (message.type === 'user') {
    return (
      <div className="flex justify-end">
        <div className="max-w-3xl bg-accent-blue text-white rounded-lg px-4 py-3">
          <p className="whitespace-pre-wrap">{message.content}</p>
        </div>
      </div>
    )
  }

  if (message.type === 'system') {
    return (
      <div className="flex justify-center">
        <div className="max-w-3xl bg-surface text-text-secondary rounded-lg px-4 py-2 text-sm">
          <p>{message.content}</p>
        </div>
      </div>
    )
  }

  // Assistant message
  return (
    <div className="flex justify-start">
      <div className="max-w-3xl">
        {/* Avatar */}
        <div className="flex items-start gap-3">
          <div className="w-8 h-8 bg-accent-blue rounded-lg flex items-center justify-center flex-shrink-0">
            <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
          </div>
          
          <div className="flex-1">
            {/* Streaming Updates */}
            {message.streamingUpdates && message.streamingUpdates.length > 0 && (
              <div className="space-y-1 mb-3">
                {message.streamingUpdates.map((update, index) => (
                  <StreamingUpdateComponent key={index} update={update} />
                ))}
              </div>
            )}
            
            {/* Main Content */}
            {message.content && (
              <div className="bg-background border border-border-light rounded-lg px-4 py-3">
                <div className="prose prose-sm max-w-none">
                  <p className="whitespace-pre-wrap">{message.content}</p>
                </div>
              </div>
            )}
            
            {/* Metadata */}
            {message.agentResponse && (
              <div className="mt-2 text-xs text-text-secondary">
                <span>
                  Processing time: {message.agentResponse.processing_time.toFixed(2)}s • 
                  Tools used: {message.agentResponse.tools_used.join(', ')} • 
                  Sources: {message.agentResponse.sources.length}
                </span>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

// Streaming Update Component
function StreamingUpdateComponent({ update }: { update: StreamingUpdate }) {
  const getIcon = () => {
    switch (update.type) {
      case 'tool':
        return (
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
          </svg>
        )
      case 'thought':
        return (
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
          </svg>
        )
      default:
        return (
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
        )
    }
  }

  return (
    <div className="flex items-center gap-2 text-sm text-text-secondary bg-surface rounded px-3 py-1.5">
      {getIcon()}
      <span>{update.content}</span>
      {update.tool && (
        <span className="text-xs bg-border-light px-2 py-0.5 rounded">
          {update.tool}
        </span>
      )}
    </div>
  )
} 