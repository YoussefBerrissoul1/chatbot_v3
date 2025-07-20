import botIcon from '../../assets/images/BotIcon.png';

const TypingIndicator = () => {
  return (
    <div className="flex justify-start animate-slide-in-up">
      <div className="flex items-start space-x-3">
        <div className="flex-shrink-0">
          <img
            src={botIcon}
            alt="Bot Icon"
            className="w-8 h-8 rounded-full object-cover border-2 border-white dark:border-gray-800 shadow-sm"
          />
        </div>
        
        <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-2xl rounded-bl-md px-4 py-3 shadow-sm max-w-xs">
          <div className="flex items-center space-x-2">
            <div className="flex space-x-1">
              <div className="w-2 h-2 bg-gray-400 dark:bg-gray-500 rounded-full animate-bounce"></div>
              <div 
                className="w-2 h-2 bg-gray-400 dark:bg-gray-500 rounded-full animate-bounce" 
                style={{ animationDelay: '0.1s' }}
              ></div>
              <div 
                className="w-2 h-2 bg-gray-400 dark:bg-gray-500 rounded-full animate-bounce" 
                style={{ animationDelay: '0.2s' }}
              ></div>
            </div>
            <span className="text-sm text-gray-500 dark:text-gray-400">
              Assistant RH tape...
            </span>
          </div>
          
          {/* Optional: Add a subtle pulse effect to the container */}
          <div className="absolute inset-0 bg-red-500/5 rounded-2xl rounded-bl-md animate-pulse opacity-50"></div>
        </div>
      </div>
    </div>
  );
};

export default TypingIndicator;

