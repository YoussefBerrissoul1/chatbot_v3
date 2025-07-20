import React, { useState, useRef, useEffect } from 'react';
import ChatHeader from './ChatHeader';
import MessageList from './MessageList';
import MessageInput from './MessageInput';
import QuickActions from './QuickActions';

// Temporary type definitions to fix import error
type Message = {
  id: string;
  content: string;
  sender: 'user' | 'bot';
  timestamp: Date;
  type: string;
};

type ChatSettings = {
  theme: string;
  fontSize: string;
  animations: boolean;
  sounds: boolean;
};

const ChatInterface: React.FC<{}> = () => {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      content: "Bonjour ! Je suis votre assistant RH Nestlé. Comment puis-je vous aider aujourd'hui ?",
      sender: 'bot',
      timestamp: new Date(),
      type: 'text'
    }
  ]);
  
  const [isTyping, setIsTyping] = useState(false);
  const [settings, setSettings] = useState<ChatSettings>({
    theme: 'light',
    fontSize: 'medium',
    animations: true,
    sounds: true
  });
  
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = async (content: string) => {
    const newMessage: Message = {
      id: Date.now().toString(),
      content,
      sender: 'user',
      timestamp: new Date(),
      type: 'text'
    };

    setMessages(prev => [...prev, newMessage]);
    setIsTyping(true);

    try {
      const res = await fetch('http://localhost:5000/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: content })
      });
      const data = await res.json();

      const botResponse: Message = {
        id: (Date.now() + 1).toString(),
        content: data.response,
        sender: 'bot',
        timestamp: new Date(),
        type: 'text'
      };

      setMessages(prev => [...prev, botResponse]);
    } catch (error) {
      setMessages(prev => [...prev, {
        id: (Date.now() + 1).toString(),
        content: "Erreur de connexion au serveur.",
        sender: 'bot',
        timestamp: new Date(),
        type: 'text'
      }]);
    }
    setIsTyping(false);
  };

  const handleQuickAction = (action: string) => {
    handleSendMessage(action);
  };

  return (
    <div className="max-w-4xl mx-auto h-screen flex flex-col bg-white dark:bg-gray-800 shadow-xl">
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
        
        <QuickActions 
          onActionClick={handleQuickAction}
          settings={settings}
        />
        
        <MessageInput 
          onSendMessage={handleSendMessage}
          settings={settings}
        />
      </div>
    </div>
  );
};

export default ChatInterface;