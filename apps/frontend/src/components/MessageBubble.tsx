import React, { useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { ChatMessage } from '../types'

interface Props {
  message: ChatMessage
}

function parseAgentResponse(content: string) {
  const sections: { type: string; content: string }[] = []
  const regex = /(Thought:|Action:|Action Input:|Observation:|Final Answer:)/g
  const parts = content.split(regex)

  for (let i = 1; i < parts.length; i += 2) {
    const type = parts[i].replace(':', '').trim()
    const contentPart = parts[i + 1]?.trim() || ''
    sections.push({ type, content: contentPart })
  }

  return sections
}

export default function MessageBubble({ message }: Props) {
  const isUser = message.role === 'user'

  if (isUser) {
    return (
      <div className={`flex justify-end`}>
        <div className="max-w-[80%] rounded-2xl px-4 py-2.5 text-sm leading-relaxed bg-blue-600 text-white rounded-br-md">
          <p className="whitespace-pre-wrap">{message.content}</p>
        </div>
      </div>
    )
  }

  const sections = parseAgentResponse(message.content)

  return (
    <div className="flex justify-start">
      <div className="max-w-[80%] rounded-2xl px-4 py-2.5 text-sm leading-relaxed bg-gray-100 text-gray-800 rounded-bl-md">
        {message.isStreaming ? (
          <p className="whitespace-pre-wrap">
            {message.content}
            <span className="inline-block w-1.5 h-4 ml-0.5 bg-blue-500 animate-pulse align-middle" />
          </p>
        ) : sections.length > 0 ? (
          <div className="space-y-2">
            {sections.map((section, idx) => (
              <div key={idx} className="border-l-2 border-blue-300 pl-3">
                <div className="font-semibold text-blue-700 text-xs uppercase tracking-wide">
                  {section.type}
                </div>
                <div className="prose prose-sm max-w-none prose-p:my-1 prose-ul:my-1 prose-ol:my-1 mt-1">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>
                    {section.content}
                  </ReactMarkdown>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="prose prose-sm max-w-none prose-p:my-1 prose-ul:my-1 prose-ol:my-1">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
              {message.content}
            </ReactMarkdown>
          </div>
        )}
      </div>
    </div>
  )
}
