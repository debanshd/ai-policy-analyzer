'use client'

import React from 'react'

interface SidebarProps {
  isCollapsed: boolean
  onToggleCollapse: () => void
  onOpenSettings?: () => void
}

export default function Sidebar({
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

 