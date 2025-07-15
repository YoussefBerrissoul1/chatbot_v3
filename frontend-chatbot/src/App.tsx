import React from 'react';
import ChatInterface from './components/ChatInterface';
import './index.css';

function App() {
  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 transition-colors duration-300">
      <ChatInterface />
    </div>
  );
}

export default App;