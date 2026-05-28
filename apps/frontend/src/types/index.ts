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

export interface StreamTokenEvent {
  type: 'token'
  content: string
}

export interface StreamRetryEvent {
  type: 'retry'
}

export interface StreamToolStartEvent {
  type: 'tool_start'
  tool: string
  input: Record<string, string>
}

export interface StreamToolEndEvent {
  type: 'tool_end'
  tool: string
  output: string
}

export interface StreamDoneEvent {
  type: 'done'
  cached?: boolean
  validation_flag?: string
  intent?: string
}

export interface StreamErrorEvent {
  type: 'error'
  message: string
}

export type StreamEvent =
  | StreamTokenEvent
  | StreamRetryEvent
  | StreamToolStartEvent
  | StreamToolEndEvent
  | StreamDoneEvent
  | StreamErrorEvent
