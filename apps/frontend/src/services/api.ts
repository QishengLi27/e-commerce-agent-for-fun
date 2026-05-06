import { ChatRequest, ChatResponse } from '../types'

const API_BASE = '/api'

export async function sendMessage(request: ChatRequest): Promise<ChatResponse> {
  const res = await fetch(`${API_BASE}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  })
  if (!res.ok) {
    throw new Error(`HTTP ${res.status}`)
  }
  return res.json()
}

export async function* streamMessage(request: ChatRequest): AsyncGenerator<string, void> {
  const res = await fetch(`${API_BASE}/chat/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  })

  if (!res.ok) {
    throw new Error(`HTTP ${res.status}`)
  }

  const reader = res.body?.getReader()
  const decoder = new TextDecoder()

  if (!reader) {
    throw new Error('No response body')
  }

  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break

    buffer += decoder.decode(value, { stream: true })
    const lines = buffer.split('\n')
    buffer = lines.pop() || ''

    for (const line of lines) {
      // SSE lines may end with \r (CRLF). Remove only the trailing \r,
      // preserve spaces inside the data field.
      const cleanLine = line.endsWith('\r') ? line.slice(0, -1) : line

      if (!cleanLine.startsWith('data:')) continue

      // Everything after "data:" — keep trailing spaces so word
      // separators sent by the backend are preserved.
      const data = cleanLine.slice(5).trimStart()

      if (data.trim() === '[DONE]') return
      yield data
    }
  }
}
