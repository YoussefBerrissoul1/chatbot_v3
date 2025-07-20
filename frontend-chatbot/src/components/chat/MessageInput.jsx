import { useState, useRef, useEffect } from 'react';
import { Send, Paperclip, Mic, Smile, Image, FileText } from 'lucide-react';

const MessageInput = ({ onSendMessage, settings }) => {
  const [message, setMessage] = useState('');
  const [isFocused, setIsFocused] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [showEmojiPicker, setShowEmojiPicker] = useState(false);
  const inputRef = useRef(null);
  const fileInputRef = useRef(null);

  const emojis = ['😊', '👍', '❤️', '😂', '🤔', '👋', '🙏', '✅', '❌', '📅', '⏰', '📧'];

  const handleSubmit = (e) => {
    e.preventDefault();
    if (message.trim()) {
      onSendMessage(message.trim());
      setMessage('');
      adjustTextareaHeight();
    }
  };

  const handleKeyPress = (e) => {
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

  const handleEmojiClick = (emoji) => {
    setMessage(prev => prev + emoji);
    setShowEmojiPicker(false);
    inputRef.current?.focus();
  };

  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      // Simulation d'upload de fichier
      onSendMessage(`📎 Fichier envoyé: ${file.name}`);
    }
  };

  const toggleRecording = () => {
    setIsRecording(!isRecording);
    if (!isRecording) {
      // Simulation d'enregistrement vocal
      setTimeout(() => {
        setIsRecording(false);
        onSendMessage("🎤 Message vocal envoyé");
      }, 3000);
    }
  };

  const quickSuggestions = [
    "Consulter mes congés",
    "Voir mes horaires",
    "Contacter le support",
    "Politique de télétravail"
  ];

  const getFontSizeClass = () => {
    switch (settings.fontSize) {
      case 'small': return 'text-sm';
      case 'large': return 'text-lg';
      default: return 'text-base';
    }
  };

  useEffect(() => {
    adjustTextareaHeight();
  }, [message]);

  return (
    <div className="bg-white dark:bg-gray-800 border-t border-gray-200 dark:border-gray-700 px-4 sm:px-6 py-4 shadow-lg">
      {/* Quick Suggestions */}
      {message === '' && (
        <div className="mb-4">
          <p className="text-xs text-gray-500 dark:text-gray-400 mb-2">Suggestions rapides :</p>
          <div className="flex flex-wrap gap-2">
            {quickSuggestions.map((suggestion, index) => (
              <button
                key={index}
                onClick={() => setMessage(suggestion)}
                className="px-3 py-1 bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-full text-xs hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors"
              >
                {suggestion}
              </button>
            ))}
          </div>
        </div>
      )}

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
              w-full px-4 py-3 pr-20 rounded-2xl resize-none
              border-2 transition-all duration-200
              bg-gray-50 dark:bg-gray-700 text-gray-900 dark:text-white
              placeholder-gray-500 dark:placeholder-gray-400
              focus:outline-none focus:ring-0
              ${getFontSizeClass()}
              ${isFocused 
                ? 'border-red-500 dark:border-red-400 bg-white dark:bg-gray-600' 
                : 'border-gray-200 dark:border-gray-600'
              }
              ${settings.animations ? 'animate-fade-in' : ''}
            `}
            style={{ minHeight: '48px', maxHeight: '120px' }}
          />
          
          {/* Input Actions */}
          <div className="absolute right-3 bottom-3 flex items-center space-x-1">
            <button
              type="button"
              onClick={() => setShowEmojiPicker(!showEmojiPicker)}
              className="p-1 rounded-full hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors"
              aria-label="Emoji"
            >
              <Smile className="w-4 h-4 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300" />
            </button>
            
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              className="p-1 rounded-full hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors"
              aria-label="Joindre un fichier"
            >
              <Paperclip className="w-4 h-4 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300" />
            </button>
            
            <button
              type="button"
              onClick={toggleRecording}
              className={`p-1 rounded-full transition-colors ${
                isRecording 
                  ? 'bg-red-100 text-red-600 animate-pulse' 
                  : 'hover:bg-gray-200 dark:hover:bg-gray-600 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300'
              }`}
              aria-label="Enregistrement vocal"
            >
              <Mic className="w-4 h-4" />
            </button>
          </div>

          {/* Emoji Picker */}
          {showEmojiPicker && (
            <div className="absolute bottom-full right-0 mb-2 bg-white dark:bg-gray-800 rounded-lg shadow-xl border border-gray-200 dark:border-gray-700 p-3">
              <div className="grid grid-cols-6 gap-2">
                {emojis.map((emoji, index) => (
                  <button
                    key={index}
                    onClick={() => handleEmojiClick(emoji)}
                    className="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded transition-colors"
                  >
                    {emoji}
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>

        <button
          type="submit"
          disabled={!message.trim() && !isRecording}
          className={`
            p-3 rounded-full transition-all duration-200 flex-shrink-0
            ${message.trim() || isRecording
              ? 'bg-gradient-to-r from-red-500 to-red-600 hover:from-red-600 hover:to-red-700 text-white shadow-lg hover:shadow-xl transform hover:scale-105'
              : 'bg-gray-200 dark:bg-gray-700 text-gray-400 dark:text-gray-500 cursor-not-allowed'
            }
          `}
          aria-label="Envoyer"
        >
          <Send className="w-5 h-5" />
        </button>
      </form>
      
      {/* Input Info */}
      <div className="flex justify-between items-center mt-2 text-xs text-gray-500 dark:text-gray-400">
        <span>
          {isRecording 
            ? "🔴 Enregistrement en cours..." 
            : "Appuyez sur Entrée pour envoyer, Shift + Entrée pour une nouvelle ligne"
          }
        </span>
        <span className={message.length > 900 ? 'text-red-500' : ''}>
          {message.length}/1000
        </span>
      </div>

      {/* Hidden file input */}
      <input
        ref={fileInputRef}
        type="file"
        onChange={handleFileUpload}
        className="hidden"
        accept=".pdf,.doc,.docx,.txt,.jpg,.jpeg,.png"
      />
    </div>
  );
};

export default MessageInput;

