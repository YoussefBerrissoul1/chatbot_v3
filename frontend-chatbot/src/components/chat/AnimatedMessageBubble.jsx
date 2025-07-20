import { motion, AnimatePresence } from 'framer-motion';
import { Check, CheckCheck, Clock, Copy, ThumbsUp, ThumbsDown } from 'lucide-react';
import { useState } from 'react';
import botIcon from '../../assets/images/BotIcon.png';

const AnimatedMessageBubble = ({ message, settings }) => {
  const [showActions, setShowActions] = useState(false);
  const [copied, setCopied] = useState(false);
  const [liked, setLiked] = useState(null); // null, 'up', 'down'
  
  const isBot = message.sender === 'bot';
  
  const bubbleClass = isBot
    ? 'bg-gradient-to-r from-red-500 to-red-600 text-white'
    : 'bg-white dark:bg-gray-800 text-gray-900 dark:text-white border border-gray-200 dark:border-gray-700';
    
  const containerClass = isBot ? 'justify-start' : 'justify-end';

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

  const handleLike = (type) => {
    setLiked(liked === type ? null : type);
  };

  const getFontSizeClass = () => {
    switch (settings.fontSize) {
      case 'small': return 'text-sm';
      case 'large': return 'text-base';
      default: return 'text-sm';
    }
  };

  const messageVariants = {
    hidden: { 
      opacity: 0, 
      y: 20,
      scale: 0.8
    },
    visible: { 
      opacity: 1, 
      y: 0,
      scale: 1,
      transition: {
        type: "spring",
        stiffness: 500,
        damping: 30
      }
    },
    exit: {
      opacity: 0,
      scale: 0.8,
      transition: {
        duration: 0.2
      }
    }
  };

  const actionsVariants = {
    hidden: { 
      opacity: 0, 
      y: 10,
      scale: 0.9
    },
    visible: { 
      opacity: 1, 
      y: 0,
      scale: 1,
      transition: {
        type: "spring",
        stiffness: 400,
        damping: 25
      }
    }
  };

  return (
    <motion.div 
      className={`flex ${containerClass} group`}
      variants={messageVariants}
      initial="hidden"
      animate="visible"
      exit="exit"
      layout
    >
      {isBot && (
        <motion.div 
          className="flex-shrink-0 mr-3"
          initial={{ scale: 0 }}
          animate={{ scale: 1 }}
          transition={{ delay: 0.2, type: "spring", stiffness: 500 }}
        >
          <img
            src={botIcon}
            alt="Bot Icon"
            className="w-8 h-8 rounded-full object-cover border-2 border-white dark:border-gray-800 shadow-sm"
          />
        </motion.div>
      )}
      
      <div className="flex flex-col max-w-xs sm:max-w-sm lg:max-w-md">
        <motion.div 
          className={`px-4 py-3 rounded-2xl shadow-sm ${bubbleClass} ${
            isBot ? 'rounded-bl-md' : 'rounded-br-md'
          }`}
          onMouseEnter={() => setShowActions(true)}
          onMouseLeave={() => setShowActions(false)}
          whileHover={{ 
            scale: 1.02,
            boxShadow: "0 10px 25px rgba(0,0,0,0.1)"
          }}
          transition={{ type: "spring", stiffness: 400 }}
        >
          <motion.p 
            className={`leading-relaxed whitespace-pre-wrap ${getFontSizeClass()}`}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.3, duration: 0.5 }}
          >
            {message.content}
          </motion.p>
          
          <motion.div 
            className={`flex items-center justify-between mt-2 text-xs ${
              isBot ? 'text-red-100' : 'text-gray-500 dark:text-gray-400'
            }`}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.5 }}
          >
            <span>
              {formatTime(message.timestamp)}
            </span>
            
            {!isBot && (
              <motion.div 
                className="flex items-center space-x-1"
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                transition={{ delay: 0.6, type: "spring" }}
              >
                {message.status === 'sent' && <Clock className="w-3 h-3" />}
                {message.status === 'delivered' && <Check className="w-3 h-3" />}
                {message.status === 'read' && <CheckCheck className="w-3 h-3 text-blue-500" />}
              </motion.div>
            )}
          </motion.div>
        </motion.div>

        {/* Animated Message Actions */}
        <AnimatePresence>
          {showActions && (
            <motion.div 
              className={`flex items-center space-x-2 mt-2 ${isBot ? 'justify-start' : 'justify-end'}`}
              variants={actionsVariants}
              initial="hidden"
              animate="visible"
              exit="hidden"
            >
              <motion.button
                onClick={copyToClipboard}
                className="p-1 rounded-full bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors"
                title="Copier le message"
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
              >
                <Copy className="w-3 h-3 text-gray-600 dark:text-gray-400" />
              </motion.button>
              
              {isBot && (
                <>
                  <motion.button
                    onClick={() => handleLike('up')}
                    className={`p-1 rounded-full transition-colors ${
                      liked === 'up' 
                        ? 'bg-green-100 text-green-600' 
                        : 'bg-gray-100 dark:bg-gray-700 hover:bg-green-100 dark:hover:bg-green-800 text-gray-600 dark:text-gray-400 hover:text-green-600'
                    }`}
                    title="Réponse utile"
                    whileHover={{ scale: 1.1 }}
                    whileTap={{ scale: 0.9 }}
                  >
                    <ThumbsUp className="w-3 h-3" />
                  </motion.button>
                  <motion.button
                    onClick={() => handleLike('down')}
                    className={`p-1 rounded-full transition-colors ${
                      liked === 'down' 
                        ? 'bg-red-100 text-red-600' 
                        : 'bg-gray-100 dark:bg-gray-700 hover:bg-red-100 dark:hover:bg-red-800 text-gray-600 dark:text-gray-400 hover:text-red-600'
                    }`}
                    title="Réponse non utile"
                    whileHover={{ scale: 1.1 }}
                    whileTap={{ scale: 0.9 }}
                  >
                    <ThumbsDown className="w-3 h-3" />
                  </motion.button>
                </>
              )}
            </motion.div>
          )}
        </AnimatePresence>

        <AnimatePresence>
          {copied && (
            <motion.div 
              className={`text-xs text-green-600 dark:text-green-400 mt-1 ${isBot ? 'text-left' : 'text-right'}`}
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              transition={{ type: "spring", stiffness: 500 }}
            >
              Message copié !
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </motion.div>
  );
};

export default AnimatedMessageBubble;

