import React from 'react'
import './globals.css'

export const metadata = {
  title: 'Policy Prism - AI-Powered Policy Analysis',
  description: 'Professional AI assistant for policy analysts and researchers',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className="font-sans">
        {children}
      </body>
    </html>
  )
} 