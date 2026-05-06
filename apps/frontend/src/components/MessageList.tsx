import { useRef, useEffect } from 'react'
import { ChatMessage } from '../types'
import MessageBubble from './MessageBubble'

interface Props {
  messages: ChatMessage[]
}

export default function MessageList({ messages }: Props) {
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  return (
    <div className="h-full overflow-y-auto chat-scroll p-4 space-y-4">
      {messages.length === 0 && (
        <div className="text-center text-gray-400 py-12">
          <p className="text-lg mb-2">👋 Welcome!</p>
          <p className="text-sm">
            Ask me anything about your orders, returns, or store policies.
          </p>
        </div>
      )}

      {messages.map((msg) => (
        <MessageBubble key={msg.id} message={msg} />
      ))}

      <div ref={bottomRef} />
    </div>
  )
}
