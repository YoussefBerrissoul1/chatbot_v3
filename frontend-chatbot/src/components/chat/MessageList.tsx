import React from 'react';
import MessageBubble from './MessageBubble';
import TypingIndicator from './TypingIndicator';
import { Message, ChatSettings } from '../types/chat';

interface MessageListProps {
  messages: Message[];
  isTyping: boolean;
  settings: ChatSettings;
}

const MessageList: React.FC<MessageListProps> = ({ messages, isTyping, settings }) => {
  const groupMessagesByDate = (messages: Message[]) => {
    const groups: { [key: string]: Message[] } = {};
    
    messages.forEach(message => {
      const date = message.timestamp.toDateString();
      if (!groups[date]) {
        groups[date] = [];
      }
      groups[date].push(message);
    });
    
    return groups;
  };

  const messageGroups = groupMessagesByDate(messages);

  return (
    <div className="flex-1 overflow-y-auto px-6 py-4 space-y-4 bg-gray-50 dark:bg-gray-900">
      {Object.entries(messageGroups).map(([date, msgs]) => (
        <div key={date}>
          <div className="flex justify-center mb-4">
            <div className="bg-white dark:bg-gray-800 px-4 py-2 rounded-full shadow-sm border border-gray-200 dark:border-gray-700">
              <span className="text-sm text-gray-500 dark:text-gray-400">
                {new Date(date).toLocaleDateString('fr-FR', { 
                  weekday: 'long', 
                  year: 'numeric', 
                  month: 'long', 
                  day: 'numeric' 
                })}
              </span>
            </div>
          </div>
          
          <div className="space-y-4">
            {msgs.map((message) => (
              <MessageBubble 
                key={message.id} 
                message={message} 
                settings={settings}
              />
            ))}
          </div>
        </div>
      ))}
      
      {isTyping && <TypingIndicator />}
    </div>
  );
};

export default MessageList;