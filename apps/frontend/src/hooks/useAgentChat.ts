import { useState, useCallback, useRef, useEffect } from 'react'
import { ChatMessage } from '../types'
import { sendMessage, streamMessage } from '../services/api'

let idCounter = 0
function nextId() {
  return `msg-${++idCounter}`
}

const SESSION_KEY = 'chat_session_id'

function getOrCreateSessionId(): string {
  let id = localStorage.getItem(SESSION_KEY)
  if (!id) {
    id = crypto.randomUUID()
    localStorage.setItem(SESSION_KEY, id)
  }
  return id
}

export function useAgentChat() {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const abortRef = useRef(false)
  const sessionIdRef = useRef<string>('')

  useEffect(() => {
    sessionIdRef.current = getOrCreateSessionId()
  }, [])

  const send = useCallback(
    async (useStream = true) => {
      const text = input.trim()
      if (!text || isLoading) return

      abortRef.current = false
      setIsLoading(true)
      setInput('')

      const userMsg: ChatMessage = {
        id: nextId(),
        role: 'user',
        content: text,
      }

      const assistantMsg: ChatMessage = {
        id: nextId(),
        role: 'assistant',
        content: '',
        isStreaming: useStream,
      }

      setMessages((prev) => [...prev, userMsg, assistantMsg])

      const sessionId = sessionIdRef.current

      try {
        if (useStream) {
          for await (const chunk of streamMessage({ message: text, session_id: sessionId })) {
            if (abortRef.current) break
            setMessages((prev) => {
              const last = prev[prev.length - 1]
              if (last.role !== 'assistant') return prev
              const updated = { ...last, content: last.content + chunk }
              return [...prev.slice(0, -1), updated]
            })
          }
        } else {
          const response = await sendMessage({ message: text, session_id: sessionId })
          setMessages((prev) => {
            const last = prev[prev.length - 1]
            if (last.role !== 'assistant') return prev
            return [
              ...prev.slice(0, -1),
              { ...last, content: response.response, isStreaming: false },
            ]
          })
        }
      } catch (err) {
        setMessages((prev) => {
          const last = prev[prev.length - 1]
          if (last.role !== 'assistant') return prev
          return [
            ...prev.slice(0, -1),
            {
              ...last,
              content: 'Sorry, something went wrong. Please try again.',
              isStreaming: false,
            },
          ]
        })
      } finally {
        setIsLoading(false)
        setMessages((prev) => {
          const last = prev[prev.length - 1]
          if (last.role !== 'assistant') return prev
          return [...prev.slice(0, -1), { ...last, isStreaming: false }]
        })
      }
    },
    [input, isLoading]
  )

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault()
        send(true)
      }
    },
    [send]
  )

  return {
    messages,
    input,
    setInput,
    isLoading,
    send,
    handleKeyDown,
  }
}
