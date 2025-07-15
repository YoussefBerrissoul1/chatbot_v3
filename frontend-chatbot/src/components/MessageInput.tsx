import React, { useState, useRef } from 'react';
import { Send, Paperclip, Mic, Smile } from 'lucide-react';
import { ChatSettings } from '../types/chat';

interface MessageInputProps {
  onSendMessage: (message: string) => void;
  settings: ChatSettings;
}

const MessageInput: React.FC<MessageInputProps> = ({ onSendMessage, settings }) => {
  const [message, setMessage] = useState('');
  const [isFocused, setIsFocused] = useState(false);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (message.trim()) {
      onSendMessage(message.trim());
      setMessage('');
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const adjustTextareaHeight = () => {
    if (inputRef.current) {
      inputRef.current.style.height = 'auto';
      inputRef.current.style.height = `${Math.min(inputRef.current.scrollHeight, 120)}px`;
    }
  };

  return (
    <div className="bg-white dark:bg-gray-800 border-t border-gray-200 dark:border-gray-700 px-6 py-4 shadow-lg">
      <form onSubmit={handleSubmit} className="flex items-end space-x-3">
        <div className="flex-1 relative">
          <textarea
            ref={inputRef}
            value={message}
            onChange={(e) => {
              setMessage(e.target.value);
              adjustTextareaHeight();
            }}
            onKeyPress={handleKeyPress}
            onFocus={() => setIsFocused(true)}
            onBlur={() => setIsFocused(false)}
            placeholder="Tapez votre message..."
            rows={1}
            className={`
              w-full px-4 py-3 pr-12 rounded-2xl resize-none
              border-2 transition-all duration-200
              bg-gray-50 dark:bg-gray-700 text-gray-900 dark:text-white
              placeholder-gray-500 dark:placeholder-gray-400
              focus:outline-none focus:ring-0
              ${isFocused 
                ? 'border-indigo-500 dark:border-indigo-400 bg-white dark:bg-gray-600' 
                : 'border-gray-200 dark:border-gray-600'
              }
              ${settings.animations ? 'animate-fade-in' : ''}
            `}
            style={{ minHeight: '48px', maxHeight: '120px' }}
          />
          
          <div className="absolute right-3 bottom-3 flex items-center space-x-2">
            <button
              type="button"
              className="p-1 rounded-full hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors"
              aria-label="Emoji"
            >
              <Smile className="w-5 h-5 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300" />
            </button>
            <button
            type="submit"
            disabled={!message.trim()}
            className={`
              p-3 rounded-full transition-all duration-200
              ${message.trim()
                ? 'bg-gradient-to-r from-indigo-500 to-purple-600 hover:from-indigo-600 hover:to-purple-700 text-white shadow-lg hover:shadow-xl transform hover:scale-105'
                : 'bg-gray-200 dark:bg-gray-700 text-gray-400 dark:text-gray-500 cursor-not-allowed'
              }
            `}
            aria-label="Envoyer"
          >
            <Send className="w-5 h-5" />
          </button>
          </div>
        </div>

        {/* <div className="flex items-center space-x-2">
          <button
            type="submit"
            disabled={!message.trim()}
            className={`
              p-3 rounded-full transition-all duration-200
              ${message.trim()
                ? 'bg-gradient-to-r from-indigo-500 to-purple-600 hover:from-indigo-600 hover:to-purple-700 text-white shadow-lg hover:shadow-xl transform hover:scale-105'
                : 'bg-gray-200 dark:bg-gray-700 text-gray-400 dark:text-gray-500 cursor-not-allowed'
              }
            `}
            aria-label="Envoyer"
          >
            <Send className="w-5 h-5" />
          </button>
        </div> */}
      </form>
      
      <div className="flex justify-between items-center mt-2 text-xs text-gray-500 dark:text-gray-400">
        <span>Appuyez sur Entrée pour envoyer, Shift + Entrée pour une nouvelle ligne</span>
        <span>{message.length}/1000</span>
      </div>
    </div>
  );
};

export default MessageInput;