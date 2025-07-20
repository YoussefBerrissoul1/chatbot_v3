import AnimatedMessageBubble from './AnimatedMessageBubble';
import TypingIndicator from './TypingIndicator';
import { motion, AnimatePresence } from 'framer-motion';

const MessageList = ({ messages, isTyping, settings }) => {
  const groupMessagesByDate = (messages) => {
    const groups = {};
    
    messages.forEach(message => {
      try {
        const date = message.timestamp instanceof Date 
          ? message.timestamp 
          : new Date(message.timestamp);
        
        if (isNaN(date.getTime())) {
          // Si la date est invalide, utiliser la date actuelle
          const today = new Date().toDateString();
          if (!groups[today]) {
            groups[today] = [];
          }
          groups[today].push(message);
        } else {
          const dateString = date.toDateString();
          if (!groups[dateString]) {
            groups[dateString] = [];
          }
          groups[dateString].push(message);
        }
      } catch (error) {
        // En cas d'erreur, grouper avec la date actuelle
        const today = new Date().toDateString();
        if (!groups[today]) {
          groups[today] = [];
        }
        groups[today].push(message);
      }
    });
    
    return groups;
  };

  const formatDateHeader = (dateString) => {
    try {
      const date = new Date(dateString);
      const today = new Date();
      const yesterday = new Date(today);
      yesterday.setDate(yesterday.getDate() - 1);

      if (date.toDateString() === today.toDateString()) {
        return "Aujourd'hui";
      } else if (date.toDateString() === yesterday.toDateString()) {
        return "Hier";
      } else {
        return date.toLocaleDateString('fr-FR', { 
          weekday: 'long', 
          year: 'numeric', 
          month: 'long', 
          day: 'numeric' 
        });
      }
    } catch (error) {
      return "Aujourd'hui";
    }
  };

  const messageGroups = groupMessagesByDate(messages);

  return (
    <div className="flex-1 overflow-y-auto px-4 sm:px-6 py-4 space-y-4 bg-gray-50 dark:bg-gray-900">
      {Object.entries(messageGroups).map(([date, msgs]) => (
        <div key={date}>
          <div className="flex justify-center mb-6">
            <div className="bg-white dark:bg-gray-800 px-4 py-2 rounded-full shadow-sm border border-gray-200 dark:border-gray-700">
              <span className="text-sm text-gray-500 dark:text-gray-400 font-medium">
                {formatDateHeader(date)}
              </span>
            </div>
          </div>
          
          <div className="space-y-4">
            {msgs.map((message) => (
              <AnimatedMessageBubble 
                key={message.id} 
                message={message} 
                settings={settings}
              />
            ))}
          </div>
        </div>
      ))}
      
      {isTyping && (
        <div className="mt-4">
          <TypingIndicator />
        </div>
      )}

      {/* Scroll to bottom indicator */}
      {messages.length > 10 && (
        <div className="flex justify-center mt-6">
          <button
            onClick={() => {
              const element = document.querySelector('.overflow-y-auto');
              element.scrollTop = element.scrollHeight;
            }}
            className="bg-white dark:bg-gray-800 text-gray-600 dark:text-gray-400 px-4 py-2 rounded-full shadow-lg border border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors text-sm"
          >
            ↓ Aller en bas
          </button>
        </div>
      )}
    </div>
  );
};

export default MessageList;

