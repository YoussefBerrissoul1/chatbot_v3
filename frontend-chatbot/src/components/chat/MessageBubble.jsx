import { Check, CheckCheck, Clock, Copy, ThumbsUp, ThumbsDown } from 'lucide-react';
import { useState } from 'react';
import botIcon from '../../assets/images/BotIcon.png';

const MessageBubble = ({ message, settings }) => {
  const [showActions, setShowActions] = useState(false);
  const [copied, setCopied] = useState(false);
  
  const isBot = message.sender === 'bot';
  
  const bubbleClass = isBot
    ? 'bg-gradient-to-r from-red-500 to-red-600 text-white'
    : 'bg-white dark:bg-gray-800 text-gray-900 dark:text-white border border-gray-200 dark:border-gray-700';
    
  const containerClass = isBot ? 'justify-start' : 'justify-end';
  
  const animationClass = settings.animations 
    ? 'animate-slide-in-up' 
    : '';

  // Fonction pour formater la date correctement
  const formatTime = (timestamp) => {
    try {
      const date = timestamp instanceof Date ? timestamp : new Date(timestamp);
      if (isNaN(date.getTime())) {
        return new Date().toLocaleTimeString('fr-FR', { 
          hour: '2-digit', 
          minute: '2-digit' 
        });
      }
      return date.toLocaleTimeString('fr-FR', { 
        hour: '2-digit', 
        minute: '2-digit' 
      });
    } catch (error) {
      return new Date().toLocaleTimeString('fr-FR', { 
        hour: '2-digit', 
        minute: '2-digit' 
      });
    }
  };

  const copyToClipboard = async () => {
    try {
      await navigator.clipboard.writeText(message.content);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Erreur lors de la copie:', err);
    }
  };

  const getFontSizeClass = () => {
    switch (settings.fontSize) {
      case 'small': return 'text-sm';
      case 'large': return 'text-base';
      default: return 'text-sm';
    }
  };

  return (
    <div className={`flex ${containerClass} ${animationClass} group`}>
      {isBot && (
        <div className="flex-shrink-0 mr-3">
          <img
            src={botIcon}
            alt="Bot Icon"
            className="w-8 h-8 rounded-full object-cover border-2 border-white dark:border-gray-800 shadow-sm"
          />
        </div>
      )}
      
      <div className="flex flex-col max-w-xs sm:max-w-sm lg:max-w-md">
        <div 
          className={`px-4 py-3 rounded-2xl shadow-sm hover:shadow-md transition-shadow duration-200 ${bubbleClass} ${
            isBot ? 'rounded-bl-md' : 'rounded-br-md'
          }`}
          onMouseEnter={() => setShowActions(true)}
          onMouseLeave={() => setShowActions(false)}
        >
          <p className={`leading-relaxed whitespace-pre-wrap ${getFontSizeClass()}`}>
            {message.content}
          </p>
          
          <div className={`flex items-center justify-between mt-2 text-xs ${
            isBot ? 'text-red-100' : 'text-gray-500 dark:text-gray-400'
          }`}>
            <span>
              {formatTime(message.timestamp)}
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

        {/* Message Actions */}
        {showActions && (
          <div className={`flex items-center space-x-2 mt-2 ${isBot ? 'justify-start' : 'justify-end'}`}>
            <button
              onClick={copyToClipboard}
              className="p-1 rounded-full bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors"
              title="Copier le message"
            >
              <Copy className="w-3 h-3 text-gray-600 dark:text-gray-400" />
            </button>
            
            {isBot && (
              <>
                <button
                  className="p-1 rounded-full bg-gray-100 dark:bg-gray-700 hover:bg-green-100 dark:hover:bg-green-800 transition-colors"
                  title="Réponse utile"
                >
                  <ThumbsUp className="w-3 h-3 text-gray-600 dark:text-gray-400 hover:text-green-600" />
                </button>
                <button
                  className="p-1 rounded-full bg-gray-100 dark:bg-gray-700 hover:bg-red-100 dark:hover:bg-red-800 transition-colors"
                  title="Réponse non utile"
                >
                  <ThumbsDown className="w-3 h-3 text-gray-600 dark:text-gray-400 hover:text-red-600" />
                </button>
              </>
            )}
          </div>
        )}

        {copied && (
          <div className={`text-xs text-green-600 dark:text-green-400 mt-1 ${isBot ? 'text-left' : 'text-right'}`}>
            Message copié !
          </div>
        )}
      </div>
    </div>
  );
};

export default MessageBubble;

