'use client'

import React, { useCallback, useState } from 'react'
import { useDropzone } from 'react-dropzone'

interface FileUploadProps {
  onFilesSelected: (files: File[]) => void
  isUploading?: boolean
  error?: string | null
}

export default function FileUpload({ onFilesSelected, isUploading = false, error }: FileUploadProps) {
  const [dragActive, setDragActive] = useState(false)

  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      onFilesSelected(acceptedFiles)
    }
  }, [onFilesSelected])

  const { getRootProps, getInputProps, isDragActive, open } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'image/*': ['.png', '.jpg', '.jpeg', '.gif', '.webp'],
    },
    multiple: true,
    disabled: isUploading,
    noClick: true, // Disable click on the root element so we can control it with the button
  })

  return (
    <div className="max-w-2xl mx-auto">
      <div
        {...getRootProps()}
        className={`upload-zone ${isDragActive ? 'active' : ''} ${isUploading ? 'opacity-50 cursor-not-allowed' : ''}`}
      >
        <input {...getInputProps()} />
        
        <div className="space-y-4">
          {/* Upload Icon */}
          <div className="mx-auto w-16 h-16 bg-blue-100 rounded-lg flex items-center justify-center">
            {isUploading ? (
              <div className="w-8 h-8 border-2 border-blue-600 border-t-transparent rounded-full animate-spin" />
            ) : (
              <svg className="w-8 h-8 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
              </svg>
            )}
          </div>
          
          {/* Upload Text */}
          <div className="text-center">
            <h3 className="text-lg font-semibold text-gray-900 mb-2">
              {isUploading ? 'Uploading files...' : 'Upload your documents'}
            </h3>
            
            {!isUploading && (
              <div className="space-y-2">
                <p className="text-gray-600">
                  {isDragActive 
                    ? 'Drop your files here'
                    : 'Drag and drop your files here, or click to browse'
                  }
                </p>
                <p className="text-sm text-gray-500">
                  Supports PDF and image files (PNG, JPG, GIF, WebP)
                </p>
              </div>
            )}
          </div>
          
          {/* Browse Button */}
          {!isUploading && (
            <div className="text-center">
              <button
                type="button"
                onClick={open}
                className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
                disabled={isUploading}
              >
                Browse Files
              </button>
            </div>
          )}
        </div>
      </div>
      
      {/* Error Message */}
      {error && (
        <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg">
          <div className="flex items-start gap-2">
            <svg className="w-5 h-5 text-red-500 mt-0.5 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <div>
              <p className="text-sm font-medium text-red-800">Upload Error</p>
              <p className="text-sm text-red-700 mt-1">{error}</p>
            </div>
          </div>
        </div>
      )}
      
      {/* Upload Guidelines */}
      <div className="mt-6 bg-gray-50 rounded-lg p-4">
        <h4 className="text-sm font-medium text-gray-900 mb-2">Upload Guidelines</h4>
        <ul className="text-sm text-gray-600 space-y-1">
          <li>• Maximum file size: 10MB per file</li>
          <li>• Supported formats: PDF, PNG, JPG, JPEG, GIF, WebP</li>
          <li>• You can upload multiple files at once</li>
          <li>• Files will be processed and indexed for analysis</li>
        </ul>
      </div>
    </div>
  )
} 