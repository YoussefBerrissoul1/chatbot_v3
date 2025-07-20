import React, { useState } from 'react';
import { Settings, Sun, Moon, Minimize2, X } from 'lucide-react';
import { ChatSettings } from '../types/chat';

interface ChatHeaderProps {
  settings: ChatSettings;
  onSettingsChange: (settings: ChatSettings) => void;
}

const ChatHeader: React.FC<ChatHeaderProps> = ({ settings, onSettingsChange }) => {
  const [showSettings, setShowSettings] = useState(false);

  const toggleTheme = () => {
    const newTheme = settings.theme === 'light' ? 'dark' : 'light';
    onSettingsChange({ ...settings, theme: newTheme });
    document.documentElement.classList.toggle('dark', newTheme === 'dark');
  };

  return (
    <header className="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 px-6 py-4 flex items-center justify-between shadow-sm">
      <div className="flex items-center space-x-4">
        <div className="relative">
          <img
            src="./public/images/logo.png"
            alt="Logo RH"
            className="w-10 h-10 rounded-full object-cover border-2 border-white dark:border-gray-800 shadow"
          />
          <div className="absolute -bottom-1 -right-1 w-4 h-4 bg-emerald-500 rounded-full border-2 border-white dark:border-gray-800">
            <div className="w-full h-full bg-emerald-500 rounded-full animate-pulse"></div>
          </div>
        </div>
        
        <div>
          <h1 className="text-lg font-semibold text-gray-900 dark:text-white">
            Assistant RH Nestlé
          </h1>
          <p className="text-sm text-emerald-600 dark:text-emerald-400 flex items-center">
            <span className="w-2 h-2 bg-emerald-500 rounded-full mr-2 animate-pulse"></span>
            En ligne - Temps de réponse ~2s
          </p>
        </div>
      </div>

      <div className="flex items-center space-x-3">
        <button
          onClick={toggleTheme}
          className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
          aria-label="Changer le thème"
        >
          {settings.theme === 'light' ? (
            <Moon className="w-5 h-5 text-gray-600 dark:text-gray-400" />
          ) : (
            <Sun className="w-5 h-5 text-gray-600 dark:text-gray-400" />
          )}
        </button>
        
        <button
          onClick={() => setShowSettings(!showSettings)}
          className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
          aria-label="Paramètres"
        >
          <Settings className="w-5 h-5 text-gray-600 dark:text-gray-400" />
        </button>
        
        <button
          className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
          aria-label="Réduire"
        >
          <Minimize2 className="w-5 h-5 text-gray-600 dark:text-gray-400" />
        </button>
        
        <button
          className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
          aria-label="Fermer"
        >
          <X className="w-5 h-5 text-gray-600 dark:text-gray-400" />
        </button>
      </div>
    </header>
  );
};

export default ChatHeader;