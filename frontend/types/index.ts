export interface FileUploadResponse {
  success: boolean;
  message: string;
  file_id: string;
  filename: string;
  size: number;
}

export interface StreamingUpdate {
  type: 'tool' | 'thought' | 'result' | 'error' | 'debug';
  content: string;
  tool?: 'RAG_Tool' | 'DataCommonsTool' | 'TavilyTool';
  metadata?: any;
}

export interface AgentResponse {
  answer: string;
  sources: string[];
  tools_used: string[];
  session_id: string;
  processing_time: number;
}

export interface UploadedFile {
  id: string;
  filename: string;
  size: number;
  uploadedAt: Date;
}

export interface AnalysisSession {
  id: string;
  name: string;
  files: UploadedFile[];
  createdAt: Date;
  lastActive: Date;
}

export interface ChatMessage {
  id: string;
  type: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: Date;
  streamingUpdates?: StreamingUpdate[];
  agentResponse?: AgentResponse;
} 