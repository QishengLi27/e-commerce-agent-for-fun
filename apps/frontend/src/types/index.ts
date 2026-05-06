export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  isStreaming?: boolean
}

export interface ChatRequest {
  session_id?: string
  message: string
}

export interface ChatResponse {
  session_id?: string
  response: string
  sources: Source[]
  cached: boolean
  latency_ms?: number
}

export interface Source {
  type: string
  id: string
  snippet: string
}
