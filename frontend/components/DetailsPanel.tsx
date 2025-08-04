'use client'

import React from 'react'
import { AnalysisSession, StreamingUpdate } from '@/types'

interface DetailsPanelProps {
  currentSession: AnalysisSession | null
  currentStreamingUpdates?: StreamingUpdate[]
  isProcessing?: boolean
  onClose: () => void
}

export default function DetailsPanel({ currentSession, currentStreamingUpdates = [], isProcessing = false, onClose }: DetailsPanelProps) {
  console.log('🗂️ DetailsPanel render:', { 
    isProcessing, 
    updatesCount: currentStreamingUpdates.length,
    showLiveProgress: isProcessing || currentStreamingUpdates.length > 0
  })
  
  return (
    <div className="h-full bg-surface p-4 flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-lg font-semibold text-text-primary">Details</h2>
        <button
          onClick={onClose}
          className="p-1 hover:bg-border-light rounded transition-colors"
          aria-label="Close details panel"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>

      {/* Live Progress Section */}
      {(isProcessing || currentStreamingUpdates.length > 0) && (
        <div className="flex-1 flex flex-col mb-6">
          <h3 className="text-sm font-medium text-text-primary mb-3 flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full ${isProcessing ? 'bg-blue-500 animate-pulse' : 'bg-green-500'}`}></div>
            Live Progress
          </h3>
          <div className="bg-background border border-border-light rounded-lg p-3 flex-1 overflow-y-auto min-h-0">
            {currentStreamingUpdates.length === 0 && !isProcessing ? (
              <p className="text-text-secondary text-sm italic">Waiting for analysis...</p>
            ) : (
              <div className="space-y-2">
                {currentStreamingUpdates.map((update, index) => (
                  <div key={index} className="flex items-start gap-2 text-sm">
                    <div className="flex-shrink-0 mt-1">
                      {getProgressIcon(update.type)}
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className={`${update.type === 'debug' ? 'text-text-tertiary text-xs font-mono' : 'text-text-secondary'}`}>
                        {update.content}
                      </p>
                      {update.tool && (
                        <span className="inline-block mt-1 text-xs bg-border-light px-2 py-0.5 rounded">
                          {update.tool}
                        </span>
                      )}
                      {update.metadata && (update.metadata.confidence !== undefined || update.metadata.retriever_time !== undefined) && (
                        <span className="text-xs text-text-tertiary ml-2">
                          {update.metadata.confidence !== undefined && `${(update.metadata.confidence * 100).toFixed(0)}%`}
                          {update.metadata.retriever_time !== undefined && `${update.metadata.retriever_time.toFixed(2)}s`}
                        </span>
                      )}
                    </div>
                  </div>
                ))}
                {isProcessing && currentStreamingUpdates.length === 0 && (
                  <div className="flex items-center gap-2 text-sm text-text-secondary">
                    <div className="w-3 h-3 border border-blue-500 border-t-transparent rounded-full animate-spin"></div>
                    <span>Starting analysis...</span>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      )}

      {!currentSession ? (
        <div className="text-center py-8">
          <div className="w-12 h-12 bg-border-light rounded-lg flex items-center justify-center mb-4 mx-auto">
            <svg className="w-6 h-6 text-text-secondary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
          </div>
          <p className="text-text-secondary text-sm">No active session</p>
        </div>
      ) : (
        <div className="flex-shrink-0">
          {/* Session Info - compact when Live Progress is not shown */}
          <div>
            <h3 className="text-sm font-medium text-text-primary mb-3">Session Information</h3>
            <div className="p-3 bg-background border border-border-light rounded-lg">
              <div className="text-sm text-text-secondary">
                <p><span className="font-medium">Session:</span> {currentSession.name}</p>
                <p><span className="font-medium">Created:</span> {formatDate(currentSession.createdAt)}</p>
                <p><span className="font-medium">Last Active:</span> {formatDate(currentSession.lastActive)}</p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Session Info when Live Progress is active - keep it compact at bottom */}
      {currentSession && (isProcessing || currentStreamingUpdates.length > 0) && (
        <div className="flex-shrink-0 mt-4 pt-4 border-t border-border-light">
          <h3 className="text-xs font-medium text-text-primary mb-2">Session</h3>
          <div className="text-xs text-text-secondary">
            <p>{currentSession.name}</p>
          </div>
        </div>
      )}
    </div>
  )
}



// Helper function to get progress icon based on update type
function getProgressIcon(type: string) {
  switch (type) {
    case 'tool':
      return (
        <svg className="w-3 h-3 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
        </svg>
      )
    case 'thought':
      return (
        <svg className="w-3 h-3 text-yellow-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
        </svg>
      )
    case 'result':
      return (
        <svg className="w-3 h-3 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
        </svg>
      )
    case 'error':
      return (
        <svg className="w-3 h-3 text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
      )
    case 'debug':
      return (
        <svg className="w-3 h-3 text-text-tertiary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.746 0 3.332.477 4.5 1.253v13C19.832 18.477 18.246 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
        </svg>
      )
    default:
      return (
        <svg className="w-3 h-3 text-text-secondary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
      )
  }
}



// Helper function to format date (simplified)
function formatDate(date: Date): string {
  return new Intl.DateTimeFormat('en-US', {
    month: 'short',
    day: 'numeric',
  }).format(date)
} 