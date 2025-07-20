import { useState, useRef, useEffect } from 'react';
import ChatHeader from './ChatHeader';
import MessageList from './MessageList';
import MessageInput from './MessageInput';
import QuickActions from './QuickActions';

const ChatInterface = () => {
  const [messages, setMessages] = useState([
    {
      id: '1',
      content: "Bonjour ! Je suis votre assistant RH Nestlé. Comment puis-je vous aider aujourd'hui ?",
      sender: 'bot',
      timestamp: new Date(),
      type: 'text'
    }
  ]);
  
  const [isTyping, setIsTyping] = useState(false);
  const [settings, setSettings] = useState({
    theme: 'light',
    fontSize: 'medium',
    animations: true,
    sounds: true
  });
  
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Remplace la simulation par un appel API réel
  const handleSendMessage = async (content) => {
    const newMessage = {
      id: Date.now().toString(),
      content,
      sender: 'user',
      timestamp: new Date(),
      type: 'text',
      status: 'sent'
    };

    setMessages(prev => [...prev, newMessage]);
    setIsTyping(true);

    try {
      const response = await fetch('http://localhost:5000/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: content })
      });
      const data = await response.json();
      const botResponse = {
        id: (Date.now() + 1).toString(),
        content: data.response || "❌ Erreur de réponse du serveur.",
        sender: 'bot',
        timestamp: new Date(),
        type: 'text'
      };
      setMessages(prev => [...prev, botResponse]);
    } catch (error) {
      setMessages(prev => [...prev, {
        id: (Date.now() + 2).toString(),
        content: "❌ Erreur de connexion au serveur.",
        sender: 'bot',
        timestamp: new Date(),
        type: 'text'
      }]);
    } finally {
      setIsTyping(false);
    }
  };

  const handleQuickAction = (action) => {
    handleSendMessage(action);
  };

  return (
    <div className="max-w-6xl mx-auto h-screen flex flex-col bg-white dark:bg-gray-900 shadow-2xl rounded-lg overflow-hidden">
      <ChatHeader 
        settings={settings} 
        onSettingsChange={setSettings}
      />
      
      <div className="flex-1 overflow-hidden flex flex-col">
        <MessageList 
          messages={messages}
          isTyping={isTyping}
          settings={settings}
        />
        
        <div ref={messagesEndRef} />
        
        {/* <QuickActions 
          onActionClick={handleQuickAction}
          settings={settings}
        /> */}
        
        <MessageInput 
          onSendMessage={handleSendMessage}
          settings={settings}
        />
      </div>
    </div>
  );
};

export default ChatInterface;

