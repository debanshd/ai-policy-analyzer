'use client'

import React from 'react'
import { AnalysisSession } from '@/types'

interface DetailsPanelProps {
  currentSession: AnalysisSession | null
  onClose: () => void
}

export default function DetailsPanel({ currentSession, onClose }: DetailsPanelProps) {
  return (
    <div className="h-full bg-surface p-4 overflow-y-auto">
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
        <div className="space-y-6">
          {/* Session Info */}
          <div>
            <h3 className="text-sm font-medium text-text-primary mb-3">Session Info</h3>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-text-secondary">Name:</span>
                <span className="text-text-primary font-medium">{currentSession.name}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-text-secondary">Created:</span>
                <span className="text-text-primary">{formatDate(currentSession.createdAt)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-text-secondary">Last Active:</span>
                <span className="text-text-primary">{formatDate(currentSession.lastActive)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-text-secondary">Session ID:</span>
                <span className="text-text-primary font-mono text-xs">{currentSession.id}</span>
              </div>
            </div>
          </div>

          {/* Uploaded Files */}
          <div>
            <h3 className="text-sm font-medium text-text-primary mb-3">
              Uploaded Files ({currentSession.files.length})
            </h3>
            
            {currentSession.files.length === 0 ? (
              <div className="text-center py-4">
                <div className="w-12 h-12 bg-border-light rounded-lg flex items-center justify-center mb-2 mx-auto">
                  <svg className="w-6 h-6 text-text-secondary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                  </svg>
                </div>
                <p className="text-text-secondary text-sm">No files uploaded</p>
              </div>
            ) : (
              <div className="space-y-2">
                {currentSession.files.map((file) => (
                  <div
                    key={file.id}
                    className="p-3 bg-background border border-border-light rounded-lg"
                  >
                    <div className="flex items-start gap-2">
                      <div className="w-5 h-5 mt-0.5 flex-shrink-0">
                        {getFileIcon(file.filename)}
                      </div>
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-medium text-text-primary truncate">
                          {file.filename}
                        </p>
                        <div className="flex items-center gap-2 mt-1 text-xs text-text-secondary">
                          <span>{formatFileSize(file.size)}</span>
                          <span>•</span>
                          <span>{formatDate(file.uploadedAt)}</span>
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Analysis Settings */}
          <div>
            <h3 className="text-sm font-medium text-text-primary mb-3">Analysis Settings</h3>
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-sm text-text-secondary">Content Icon</span>
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                  <span className="text-sm text-text-primary">Active</span>
                </div>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm text-text-secondary">Data Commons</span>
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                  <span className="text-sm text-text-primary">Connected</span>
                </div>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm text-text-secondary">Web Search</span>
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                  <span className="text-sm text-text-primary">Available</span>
                </div>
              </div>
            </div>
          </div>

          {/* Suggested Follow-up Questions */}
          <div>
            <h3 className="text-sm font-medium text-text-primary mb-3">Suggested Follow-up Questions</h3>
            <div className="space-y-2">
              {getSuggestedQuestions(currentSession).map((question, index) => (
                <button
                  key={index}
                  className="w-full text-left p-2 text-sm text-text-secondary hover:text-text-primary hover:bg-background rounded transition-colors"
                >
                  {question}
                </button>
              ))}
            </div>
          </div>

          {/* Export Options */}
          <div>
            <h3 className="text-sm font-medium text-text-primary mb-3">Export Options</h3>
            <div className="space-y-2">
              <button className="w-full btn-secondary text-sm py-2">
                Export Session
              </button>
              <button className="w-full btn-secondary text-sm py-2">
                Save as PDF
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

// Helper function to get file icon based on filename
function getFileIcon(filename: string) {
  const extension = filename.toLowerCase().split('.').pop()
  
  if (extension === 'pdf') {
    return (
      <svg className="w-5 h-5 text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
      </svg>
    )
  } else if (['png', 'jpg', 'jpeg', 'gif', 'webp'].includes(extension || '')) {
    return (
      <svg className="w-5 h-5 text-purple-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
      </svg>
    )
  } else {
    return (
      <svg className="w-5 h-5 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
      </svg>
    )
  }
}

// Helper function to format file size
function formatFileSize(bytes: number): string {
  if (bytes === 0) return '0 B'
  const k = 1024
  const sizes = ['B', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(1))} ${sizes[i]}`
}

// Helper function to format date
function formatDate(date: Date): string {
  return new Intl.DateTimeFormat('en-US', {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  }).format(date)
}

// Helper function to get suggested questions based on session
function getSuggestedQuestions(session: AnalysisSession): string[] {
  if (session.files.length === 0) {
    return [
      'Upload documents to get started',
      'What types of files can I analyze?',
    ]
  }
  
  return [
    'What are the main findings in these documents?',
    'Summarize the key recommendations',
    'How does this relate to current policy trends?',
    'What are the economic implications?',
    'Compare findings across uploaded documents',
  ]
} 