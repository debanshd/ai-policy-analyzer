'use client'

import React, { useState, useEffect } from 'react'

interface SettingsModalProps {
  isOpen: boolean
  onClose: () => void
  onSave: (keys: ApiKeys) => Promise<void>
}

interface ApiKeys {
  openai_api_key: string
  tavily_api_key: string
  data_commons_api_key: string
  langsmith_api_key: string
}

export default function SettingsModal({ isOpen, onClose, onSave }: SettingsModalProps) {
  const [keys, setKeys] = useState<ApiKeys>({
    openai_api_key: '',
    tavily_api_key: '',
    data_commons_api_key: '',
    langsmith_api_key: '',
  })
  const [saving, setSaving] = useState(false)
  const [showKeys, setShowKeys] = useState<Record<string, boolean>>({})

  useEffect(() => {
    if (isOpen) {
      // Load existing keys from backend when modal opens
      loadExistingKeys()
    }
  }, [isOpen])

  const loadExistingKeys = async () => {
    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/settings`)
      if (response.ok) {
        const data = await response.json()
        console.log('Loaded keys:', data)
        
        // Ensure all keys are strings to avoid controlled/uncontrolled input issues
        const loadedKeys = data.api_keys || {}
        setKeys({
          openai_api_key: loadedKeys.openai_api_key || '',
          tavily_api_key: loadedKeys.tavily_api_key || '',
          data_commons_api_key: loadedKeys.data_commons_api_key || '',
          langsmith_api_key: loadedKeys.langsmith_api_key || '',
        })
      }
    } catch (error) {
      console.error('Failed to load existing keys:', error)
    }
  }

  const handleSave = async () => {
    setSaving(true)
    try {
      await onSave(keys)
      onClose()
    } catch (error) {
      console.error('Failed to save API keys:', error)
      alert('Failed to save API keys. Please try again.')
    } finally {
      setSaving(false)
    }
  }

  const toggleShowKey = (keyName: string) => {
    setShowKeys(prev => ({
      ...prev,
      [keyName]: !prev[keyName]
    }))
  }

  const handleInputChange = (keyName: keyof ApiKeys, value: string) => {
    setKeys(prev => ({
      ...prev,
      [keyName]: value
    }))
  }

  const getKeyDescription = (keyName: string) => {
    switch (keyName) {
      case 'openai_api_key':
        return 'Required for document analysis and AI responses'
      case 'tavily_api_key':
        return 'Optional - for web search capabilities'
      case 'data_commons_api_key':
        return 'Optional - for statistical data integration'
      case 'langsmith_api_key':
        return 'Optional - for monitoring and debugging'
      default:
        return ''
    }
  }

  const isRequired = (keyName: string) => {
    return keyName === 'openai_api_key'
  }

  const getProviderUrl = (keyName: string) => {
    switch (keyName) {
      case 'openai_api_key':
        return 'https://platform.openai.com/api-keys'
      case 'tavily_api_key':
        return 'https://tavily.com'
      case 'data_commons_api_key':
        return 'https://datacommons.org/api'
      case 'langsmith_api_key':
        return 'https://smith.langchain.com/'
      default:
        return ''
    }
  }

  const getDisplayName = (keyName: string) => {
    switch (keyName) {
      case 'openai_api_key':
        return 'OpenAI API Key'
      case 'tavily_api_key':
        return 'Tavily API Key'
      case 'data_commons_api_key':
        return 'Data Commons API Key'
      case 'langsmith_api_key':
        return 'LangSmith API Key'
      default:
        return keyName.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())
    }
  }

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-lg shadow-xl max-w-2xl w-full max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="border-b border-gray-200 p-6">
          <div className="flex items-center justify-between">
            <h2 className="text-xl font-semibold text-gray-900">API Key Settings</h2>
            <button
              onClick={onClose}
              className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
              aria-label="Close settings"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
          <p className="text-gray-600 mt-2">
            Configure your API keys to enable Policy Prism's analysis features
          </p>
        </div>

        {/* Content */}
        <div className="p-6 space-y-6">

          {/* OpenAI API Key */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <label className="block text-sm font-medium text-gray-900">
                OpenAI API Key <span className="text-red-500 ml-1">*</span>
              </label>
              <a
                href="https://platform.openai.com/api-keys"
                target="_blank"
                rel="noopener noreferrer"
                className="text-xs text-blue-600 hover:underline"
              >
                Get API Key
              </a>
            </div>
            <div className="relative">
              <input
                type={showKeys['openai_api_key'] ? "text" : "password"}
                value={keys.openai_api_key}
                onChange={(e) => handleInputChange('openai_api_key', e.target.value)}
                placeholder="Enter your OpenAI API Key here..."
                className="w-full px-3 py-2 pr-10 border border-gray-300 rounded-lg focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 bg-white text-gray-900"
              />
              <button
                type="button"
                onClick={() => toggleShowKey('openai_api_key')}
                className="absolute right-2 top-1/2 transform -translate-y-1/2 p-1 hover:bg-gray-100 rounded"
              >
                {showKeys['openai_api_key'] ? '👁️' : '🙈'}
              </button>
            </div>
            <p className="text-xs text-gray-500">
              Required for document analysis and AI responses
            </p>
          </div>

          {/* Tavily API Key */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <label className="block text-sm font-medium text-gray-900">
                Tavily API Key
              </label>
              <a
                href="https://tavily.com"
                target="_blank"
                rel="noopener noreferrer"
                className="text-xs text-blue-600 hover:underline"
              >
                Get API Key
              </a>
            </div>
            <div className="relative">
              <input
                type={showKeys['tavily_api_key'] ? "text" : "password"}
                value={keys.tavily_api_key}
                onChange={(e) => handleInputChange('tavily_api_key', e.target.value)}
                placeholder="Enter your Tavily API Key here..."
                className="w-full px-3 py-2 pr-10 border border-gray-300 rounded-lg focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 bg-white text-gray-900"
              />
              <button
                type="button"
                onClick={() => toggleShowKey('tavily_api_key')}
                className="absolute right-2 top-1/2 transform -translate-y-1/2 p-1 hover:bg-gray-100 rounded"
              >
                {showKeys['tavily_api_key'] ? '👁️' : '🙈'}
              </button>
            </div>
            <p className="text-xs text-gray-500">
              Optional - for web search capabilities
            </p>
          </div>

          {/* Data Commons API Key */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <label className="block text-sm font-medium text-gray-900">
                Data Commons API Key
              </label>
              <a
                href="https://datacommons.org/api"
                target="_blank"
                rel="noopener noreferrer"
                className="text-xs text-blue-600 hover:underline"
              >
                Get API Key
              </a>
            </div>
            <div className="relative">
              <input
                type={showKeys['data_commons_api_key'] ? "text" : "password"}
                value={keys.data_commons_api_key}
                onChange={(e) => handleInputChange('data_commons_api_key', e.target.value)}
                placeholder="Enter your Data Commons API Key here..."
                className="w-full px-3 py-2 pr-10 border border-gray-300 rounded-lg focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 bg-white text-gray-900"
              />
              <button
                type="button"
                onClick={() => toggleShowKey('data_commons_api_key')}
                className="absolute right-2 top-1/2 transform -translate-y-1/2 p-1 hover:bg-gray-100 rounded"
              >
                {showKeys['data_commons_api_key'] ? '👁️' : '🙈'}
              </button>
            </div>
            <p className="text-xs text-gray-500">
              Optional - for statistical data integration
            </p>
          </div>

          {/* LangSmith API Key */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <label className="block text-sm font-medium text-gray-900">
                LangSmith API Key
              </label>
              <a
                href="https://smith.langchain.com/"
                target="_blank"
                rel="noopener noreferrer"
                className="text-xs text-blue-600 hover:underline"
              >
                Get API Key
              </a>
            </div>
            <div className="relative">
              <input
                type={showKeys['langsmith_api_key'] ? "text" : "password"}
                value={keys.langsmith_api_key}
                onChange={(e) => handleInputChange('langsmith_api_key', e.target.value)}
                placeholder="Enter your LangSmith API Key here..."
                className="w-full px-3 py-2 pr-10 border border-gray-300 rounded-lg focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 bg-white text-gray-900"
              />
              <button
                type="button"
                onClick={() => toggleShowKey('langsmith_api_key')}
                className="absolute right-2 top-1/2 transform -translate-y-1/2 p-1 hover:bg-gray-100 rounded"
              >
                {showKeys['langsmith_api_key'] ? '👁️' : '🙈'}
              </button>
            </div>
            <p className="text-xs text-gray-500">
              Optional - for monitoring and debugging
            </p>
          </div>

          {/* Information Box */}
          <div className="bg-blue-50 rounded-lg p-4">
            <div className="flex items-start gap-3">
              <svg className="w-5 h-5 text-blue-600 mt-0.5 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <div>
                <h4 className="text-sm font-medium text-blue-800 mb-1">Security Note</h4>
                <p className="text-xs text-gray-600">
                  Your API keys are stored securely in memory only and are never persisted to disk. 
                  You'll need to re-enter them if you restart the application.
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="border-t border-gray-200 p-6 flex justify-end space-x-3">
          <button
            onClick={onClose}
            className="px-4 py-2 border border-gray-300 rounded-lg text-gray-700 hover:bg-gray-50 transition-colors"
            disabled={saving}
          >
            Cancel
          </button>
          <button
            onClick={handleSave}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            disabled={saving || !keys.openai_api_key}
          >
            {saving ? (
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                Saving...
              </div>
            ) : (
              'Save Settings'
            )}
          </button>
        </div>
      </div>
    </div>
  )
} 