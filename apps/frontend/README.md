# Frontend App (Not Yet Implemented)

## Overview

This directory will contain the customer-facing web interface for the e-commerce support agent.

## Planned Tech Stack

| Concern | Choice | Rationale |
|---------|--------|-----------|
| Framework | React 18+ or Vue 3 | Component-driven, large ecosystem |
| Language | TypeScript | Type safety shared with `packages/shared` |
| Styling | Tailwind CSS | Rapid UI development, consistent design system |
| State | TanStack Query (React Query) | Server state caching, background refetch |
| Real-time | Socket.IO client or EventSource | Streaming agent responses |
| Build tool | Vite | Fast HMR, modern bundling |

## Planned Features

1. **Chat Widget**
   - Floating chat bubble on all pages
   - Expandable full-page chat view
   - Typing indicators and streaming text

2. **Conversation History**
   - Sidebar with past sessions
   - Searchable history

3. **Rich Message Rendering**
   - Markdown support for policy excerpts
   - Clickable order status cards
   - Source citations with expand/collapse

4. **Feedback**
   - Thumbs up/down per message
   - Optional free-text follow-up on negative feedback

5. **Accessibility**
   - Keyboard navigation
   - ARIA labels
   - Reduced-motion support

## Folder Structure (Planned)

```
apps/frontend/
├── public/
├── src/
│   ├── components/
│   │   ├── ChatWidget/
│   │   ├── MessageList/
│   │   ├── MessageBubble/
│   │   └── SourceCitation/
│   ├── hooks/
│   │   ├── useChat.ts
│   │   └── useStreaming.ts
│   ├── services/
│   │   └── api.ts
│   ├── types/
│   │   └── index.ts          # from packages/shared
│   └── App.tsx
├── index.html
├── package.json
├── tailwind.config.js
├── tsconfig.json
└── vite.config.ts
```

## API Contract (Draft)

The frontend will communicate with `apps/backend/api/` via:

- `POST /chat` — for standard request/response
- `WebSocket /chat/stream` — for token-by-token streaming

## Getting Started (Future)

```bash
cd apps/frontend
npm install
npm run dev
```
