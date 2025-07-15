import React from 'react';
import { Check, CheckCheck, Clock } from 'lucide-react';
import { Message, ChatSettings } from '../types/chat';

interface MessageBubbleProps {
  message: Message;
  settings: ChatSettings;
}

const MessageBubble: React.FC<MessageBubbleProps> = ({ message, settings }) => {
  const isBot = message.sender === 'bot';
  
  const bubbleClass = isBot
    ? 'bg-gradient-to-r from-indigo-500 to-purple-600 text-white'
    : 'bg-white dark:bg-gray-800 text-gray-900 dark:text-white border border-gray-200 dark:border-gray-700';
    
  const containerClass = isBot ? 'justify-start' : 'justify-end';
  
  const animationClass = settings.animations 
    ? 'animate-slide-in-up' 
    : '';

  return (
    <div className={`flex ${containerClass} ${animationClass}`}>
      {isBot && (
        <img
          src="./public/images/logo.png"
          alt="Logo RH"
          className="w-8 h-8 rounded-full object-cover border-2 border-white dark:border-gray-800 shadow mr-2 self-end"
        />
      )}
      <div className={`max-w-xs lg:max-w-md px-4 py-3 rounded-2xl shadow-sm ${bubbleClass} ${isBot ? 'rounded-bl-md' : 'rounded-br-md'}`}>
        <p className="text-sm leading-relaxed">{message.content}</p>
        
        <div className={`flex items-center justify-between mt-2 text-xs ${isBot ? 'text-indigo-100' : 'text-gray-500 dark:text-gray-400'}`}>
          <span>
            {message.timestamp.toLocaleTimeString('fr-FR', { 
              hour: '2-digit', 
              minute: '2-digit' 
            })}
          </span>
          
          {!isBot && (
            <div className="flex items-center space-x-1">
              {message.status === 'sent' && <Clock className="w-3 h-3" />}
              {message.status === 'delivered' && <Check className="w-3 h-3" />}
              {message.status === 'read' && <CheckCheck className="w-3 h-3 text-blue-500" />}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default MessageBubble;