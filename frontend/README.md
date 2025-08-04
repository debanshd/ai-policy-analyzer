# Policy Prism Frontend

A professional Next.js frontend for the AI-powered policy analysis application.

## Features

- **Clean Three-Panel Layout**: Left sidebar for navigation, main panel for interaction, right details panel
- **Professional Design**: Inspired by Notion and Linear with a clean, minimal aesthetic  
- **File Upload**: Drag-and-drop support for PDF and image files
- **Real-time Streaming**: Live display of AI agent thought process and results
- **Session Management**: Persistent analysis sessions with history
- **Responsive Design**: Optimized for desktop and tablet usage

## Quick Start

1. **Install Dependencies**
   ```bash
   npm install
   ```

2. **Start Development Server**
   ```bash
   npm run dev
   ```

3. **Access Application**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000 (must be running separately)

## Usage

1. **Create New Analysis**: Click "New Analysis" in the sidebar
2. **Upload Documents**: Drag and drop PDF or image files into the upload zone
3. **Ask Questions**: Use the chat interface to analyze your documents
4. **View Results**: Watch the real-time streaming of AI analysis process
5. **Review Details**: Check the right panel for session info and suggested questions

## Architecture

- **Framework**: Next.js 14 with App Router
- **Styling**: Tailwind CSS with custom design tokens
- **State Management**: React hooks for local state
- **File Upload**: react-dropzone for drag-and-drop functionality
- **Streaming**: Native fetch with ReadableStream for SSE
- **TypeScript**: Full type safety throughout the application

## Design Philosophy

The UI follows a **Professional, Clean, and Data-First** approach:

- **Minimalist**: Clean interface without distractions
- **Structured**: Clear visual hierarchy and organization
- **Responsive**: Smooth animations and interactions
- **Transparent**: Shows AI reasoning process in real-time
- **Efficient**: Optimized for focused analytical work

## Color Palette

- **Background**: #FFFFFF (clean white)
- **Surface**: #F9FAFB (light gray backgrounds)
- **Primary Text**: #111827 (dark gray)
- **Secondary Text**: #6B7280 (lighter gray)
- **Accent**: #3B82F6 (calm blue)
- **Borders**: #E5E7EB and #F3F4F6

## Development

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Start production server
npm start

# Run linting
npm run lint
```

## Integration with Backend

The frontend expects the backend API to be running on `http://localhost:8000` by default. Key endpoints used:

- `POST /upload` - File upload with session management
- `POST /query` - Streaming query processing
- `GET /session/{id}` - Session information retrieval

## Browser Support

- Chrome/Edge 90+
- Firefox 88+
- Safari 14+

## Contributing

1. Follow the existing code style and structure
2. Use TypeScript for all new components
3. Maintain the clean, professional design aesthetic
4. Test streaming functionality thoroughly
5. Keep components focused and reusable 