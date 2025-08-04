'use client'

import React from 'react'
import { AnalysisSession } from '@/types'

interface SidebarProps {
  sessions: AnalysisSession[]
  currentSession: AnalysisSession | null
  onNewSession: () => void
  onSelectSession: (session: AnalysisSession) => void
  isCollapsed: boolean
  onToggleCollapse: () => void
  onOpenSettings?: () => void
}

export default function Sidebar({
  sessions,
  currentSession,
  onNewSession,
  onSelectSession,
  isCollapsed,
  onToggleCollapse,
  onOpenSettings,
}: SidebarProps) {
  if (isCollapsed) {
    return (
      <div className="sidebar w-16 p-2 flex flex-col items-center">
        <button
          onClick={onToggleCollapse}
          className="p-2 hover:bg-border-light rounded-lg transition-colors mb-4"
          aria-label="Expand sidebar"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
          </svg>
        </button>
        
        <button
          onClick={onNewSession}
          className="p-2 hover:bg-border-light rounded-lg transition-colors"
          aria-label="New analysis"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
          </svg>
        </button>
      </div>
    )
  }

  return (
    <div className="sidebar p-4">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 bg-accent-blue rounded-lg flex items-center justify-center">
            <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
          </div>
          <h1 className="text-lg font-semibold text-text-primary">Policy Prism</h1>
        </div>
        
        <button
          onClick={onToggleCollapse}
          className="p-1 hover:bg-border-light rounded transition-colors"
          aria-label="Collapse sidebar"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 19l-7-7 7-7m8 14l-7-7 7-7" />
          </svg>
        </button>
      </div>

      {/* New Analysis Button */}
      <button
        onClick={onNewSession}
        className="w-full btn-primary mb-6 flex items-center justify-center gap-2"
      >
        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
        </svg>
        New Analysis
      </button>

      {/* Session History */}
      <div className="mb-6">
        <h2 className="text-sm font-medium text-text-secondary mb-3">Past Analysis</h2>
        <div className="space-y-1">
          {sessions.map((session) => (
            <button
              key={session.id}
              onClick={() => onSelectSession(session)}
              className={`w-full text-left p-3 rounded-lg transition-colors ${
                currentSession?.id === session.id
                  ? 'bg-accent-blue-light text-accent-blue'
                  : 'hover:bg-border-light text-text-primary'
              }`}
            >
              <div className="flex items-start gap-2">
                <div className="w-5 h-5 mt-0.5 flex-shrink-0">
                  {session.files.length > 0 ? (
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                    </svg>
                  ) : (
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 8h10M7 12h4m1 8l-4-4H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-3l-4 4z" />
                    </svg>
                  )}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="text-sm font-medium truncate">{session.name}</div>
                  <div className="text-xs text-text-secondary">
                    {session.files.length} file{session.files.length !== 1 ? 's' : ''} • {' '}
                    {formatRelativeTime(session.lastActive)}
                  </div>
                </div>
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* Footer Links */}
      <div className="mt-auto pt-4 border-t border-border-light">
        <div className="space-y-2">
          <button 
            onClick={onOpenSettings}
            className="w-full text-left text-sm text-text-secondary hover:text-text-primary transition-colors py-1"
          >
            Settings
          </button>
          <button className="w-full text-left text-sm text-text-secondary hover:text-text-primary transition-colors py-1">
            Help & Support
          </button>
        </div>
      </div>
    </div>
  )
}

// Helper function to format relative time
function formatRelativeTime(date: Date): string {
  const now = new Date()
  const diffInSeconds = Math.floor((now.getTime() - date.getTime()) / 1000)
  
  if (diffInSeconds < 60) return 'Just now'
  if (diffInSeconds < 3600) return `${Math.floor(diffInSeconds / 60)}m ago`
  if (diffInSeconds < 86400) return `${Math.floor(diffInSeconds / 3600)}h ago`
  if (diffInSeconds < 604800) return `${Math.floor(diffInSeconds / 86400)}d ago`
  
  return date.toLocaleDateString()
} 