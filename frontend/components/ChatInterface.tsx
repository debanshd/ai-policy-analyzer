'use client'

import React, { useState, useRef, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { ChatMessage } from '@/types'

interface ChatInterfaceProps {
  messages: ChatMessage[]
  onSendMessage: (content: string) => Promise<void>
  disabled?: boolean
}

// Function to highlight source citations with different colors
function highlightSourceCitations(text: string): React.ReactNode[] {
  if (typeof text !== 'string') return [text]
  
  // Citation patterns with different colors
  const patterns = [
    {
      // Data Commons citations: [Data Commons: URL] or [Data Commons]
      regex: /\[Data Commons(?:: ([^\]]+))?\]/g,
      color: 'bg-green-100 text-green-800 border border-green-200',
      icon: '📊'
    },
    {
      // Web/Tavily citations: [Web: URL] or [Web] or [Web Search: URL]
      regex: /\[(?:Web|Web Search)(?:: ([^\]]+))?\]/g,
      color: 'bg-blue-100 text-blue-800 border border-blue-200',
      icon: '🌐'
    },
    {
      // PDF citations: [PDF: filename] or [PDF]
      regex: /\[PDF(?:: ([^\]]+))?\]/g,
      color: 'bg-purple-100 text-purple-800 border border-purple-200',
      icon: '📄'
    }
  ]
  
  let parts: React.ReactNode[] = []
  let lastIndex = 0
  let matchFound = false
  
  // Find all matches across all patterns
  const allMatches: Array<{
    match: RegExpMatchArray
    pattern: typeof patterns[0]
    index: number
  }> = []
  
  patterns.forEach(pattern => {
    let match
    const regex = new RegExp(pattern.regex.source, 'g')
    while ((match = regex.exec(text)) !== null) {
      const matchIndex = match.index ?? 0
      allMatches.push({
        match,
        pattern,
        index: matchIndex
      })
    }
  })
  
  // Sort matches by position
  allMatches.sort((a, b) => a.index - b.index)
  
  // Process matches
  allMatches.forEach(({ match, pattern, index }, i) => {
    matchFound = true
    
    // Add text before match
    if (index > lastIndex) {
      parts.push(text.slice(lastIndex, index))
    }
    
    // Create highlighted citation
    const fullMatch = match[0]
    const url = match[1]
    
    const citationElement = (
      <span
        key={`citation-${i}`}
        className={`inline-flex items-center gap-1 px-2 py-1 rounded-md text-xs font-medium ${pattern.color}`}
        title={url || fullMatch}
      >
        <span>{pattern.icon}</span>
        <span>{fullMatch}</span>
        {url && (
          <a
            href={url}
            target="_blank"
            rel="noopener noreferrer"
            className="ml-1 underline hover:no-underline"
            onClick={(e) => e.stopPropagation()}
          >
            🔗
          </a>
        )}
      </span>
    )
    
    parts.push(citationElement)
    lastIndex = index + fullMatch.length
  })
  
  // Add remaining text
  if (lastIndex < text.length) {
    parts.push(text.slice(lastIndex))
  }
  
  // If no matches found, return original text
  return matchFound ? parts : [text]
}

export default function ChatInterface({ messages, onSendMessage, disabled = false }: ChatInterfaceProps) {
  const [inputValue, setInputValue] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)



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
            {/* Streaming Updates removed - now only shown in right panel */}
            
            {/* Main Content */}
            {message.content && (
              <div className="bg-background border border-border-light rounded-lg px-4 py-3">
                <div className="prose prose-sm max-w-none">
                  <ReactMarkdown 
                    remarkPlugins={[remarkGfm]}
                    components={{
                      // Custom link component to make external links open in new tab
                      a: ({ href, children, ...props }) => (
                        <a 
                          href={href} 
                          target={href?.startsWith('http') ? '_blank' : undefined}
                          rel={href?.startsWith('http') ? 'noopener noreferrer' : undefined}
                          className="text-accent-blue hover:text-accent-blue-dark underline"
                          {...props}
                        >
                          {children}
                        </a>
                      ),
                      // Enhanced paragraph component with source highlighting
                      p: ({ children }) => {
                        const processChildren = (child: any): any => {
                          if (typeof child === 'string') {
                            return highlightSourceCitations(child)
                          }
                          return child
                        }
                        
                        return (
                          <p className="mb-3 last:mb-0 leading-relaxed">
                            {React.Children.map(children, processChildren)}
                          </p>
                        )
                      },
                      // Enhanced heading components
                      h1: ({ children }) => (
                        <h1 className="text-xl font-bold mb-3 mt-4 first:mt-0 text-text-primary">{children}</h1>
                      ),
                      h2: ({ children }) => (
                        <h2 className="text-lg font-semibold mb-2 mt-3 first:mt-0 text-text-primary">{children}</h2>
                      ),
                      h3: ({ children }) => (
                        <h3 className="text-base font-medium mb-2 mt-3 first:mt-0 text-text-primary">{children}</h3>
                      ),
                      // Custom list components with better spacing
                      ul: ({ children }) => (
                        <ul className="list-disc list-inside mb-3 space-y-1 pl-2">{children}</ul>
                      ),
                      ol: ({ children }) => (
                        <ol className="list-decimal list-inside mb-3 space-y-1 pl-2">{children}</ol>
                      ),
                      li: ({ children }) => (
                        <li className="mb-1">
                          {React.Children.map(children, child => 
                            typeof child === 'string' ? highlightSourceCitations(child) : child
                          )}
                        </li>
                      ),
                      // Enhanced code blocks
                      code: ({ children, ...props }: any) => {
                        const isInline = !String(children).includes('\n')
                        return isInline ? (
                          <code className="bg-gray-100 px-1.5 py-0.5 rounded text-sm font-mono" {...props}>
                            {children}
                          </code>
                        ) : (
                          <pre className="bg-gray-100 p-4 rounded-lg overflow-x-auto mb-3">
                            <code className="text-sm font-mono" {...props}>
                              {children}
                            </code>
                          </pre>
                        )
                      },
                      // Enhanced blockquote
                      blockquote: ({ children }) => (
                        <blockquote className="border-l-4 border-gray-300 pl-4 italic mb-3 text-text-secondary">
                          {children}
                        </blockquote>
                      )
                    }}
                  >
                    {message.content}
                  </ReactMarkdown>
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

// Streaming updates now only shown in right panel DetailsPanel component 