import ChatWidget from './components/ChatWidget'

function App() {
  return (
    <div className="min-h-screen bg-gray-50 flex items-center justify-center p-4">
      <div className="max-w-2xl w-full">
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">
            Smart E-Commerce Support
          </h1>
          <p className="text-gray-600">
            Ask about orders, returns, shipping, and policies.
          </p>
        </div>
        <ChatWidget />
      </div>
    </div>
  )
}

export default App
