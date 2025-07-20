import { useState } from 'react';
import { Settings, Sun, Moon, Minimize2, X, ArrowLeft } from 'lucide-react';
import { Link, useNavigate } from 'react-router-dom';
import botIcon from '../../assets/images/BotIcon.png';

const ChatHeader = ({ settings, onSettingsChange }) => {
  const [showSettings, setShowSettings] = useState(false);
  const navigate = useNavigate();

  const toggleTheme = () => {
    const newTheme = settings.theme === 'light' ? 'dark' : 'light';
    onSettingsChange({ ...settings, theme: newTheme });
    document.documentElement.classList.toggle('dark', newTheme === 'dark');
  };

  const toggleAnimations = () => {
    onSettingsChange({ ...settings, animations: !settings.animations });
  };

  const toggleSounds = () => {
    onSettingsChange({ ...settings, sounds: !settings.sounds });
  };

  const changeFontSize = (size) => {
    onSettingsChange({ ...settings, fontSize: size });
  };

  return (
    <>
      <header className="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 px-4 sm:px-6 py-4 flex items-center justify-between shadow-sm relative">
        <div className="flex items-center space-x-4">
          {/* Back to Landing Page */}
          <Link 
            to="/"
            className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors lg:hidden"
            aria-label="Retour à l'accueil"
          >
            <ArrowLeft className="w-5 h-5 text-gray-600 dark:text-gray-400" />
          </Link>

          <div className="relative">
            <img
              src={botIcon}
              alt="Bot Icon"
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
              En ligne 
            </p>
          </div>
        </div>

        <div className="flex items-center space-x-2 sm:space-x-3">
          <Link 
            to="/"
            className="hidden lg:flex items-center px-3 py-2 text-sm font-medium text-gray-700 dark:text-gray-300 hover:text-red-600 dark:hover:text-red-400 transition-colors"
          >
            <ArrowLeft className="w-4 h-4 mr-1" />
            Accueil
          </Link>

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
          
          {/*
          <button
            className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors hidden sm:block"
            aria-label="Réduire"
          >
            <Minimize2 className="w-5 h-5 text-gray-600 dark:text-gray-400" />
          </button>
          */}
          
          <button
            className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors hidden sm:block"
            aria-label="Fermer"
            onClick={() => navigate('/')}
          >
            <X className="w-5 h-5 text-gray-600 dark:text-gray-400" />
          </button>
        </div>

        {/* Settings Panel */}
        {showSettings && (
          <div className="absolute top-full right-4 mt-2 w-80 bg-white dark:bg-gray-800 rounded-lg shadow-xl border border-gray-200 dark:border-gray-700 z-50">
            <div className="p-6">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                Paramètres
              </h3>
              
              {/* Theme Toggle */}
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Thème
                </label>
                <button
                  onClick={toggleTheme}
                  className="w-full flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-700 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-600 transition-colors"
                >
                  <span className="text-gray-900 dark:text-white">
                    {settings.theme === 'light' ? 'Clair' : 'Sombre'}
                  </span>
                  {settings.theme === 'light' ? (
                    <Sun className="w-5 h-5 text-gray-600 dark:text-gray-400" />
                  ) : (
                    <Moon className="w-5 h-5 text-gray-600 dark:text-gray-400" />
                  )}
                </button>
              </div>

              {/* Font Size */}
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Taille de police
                </label>
                <div className="grid grid-cols-3 gap-2">
                  {['small', 'medium', 'large'].map((size) => (
                    <button
                      key={size}
                      onClick={() => changeFontSize(size)}
                      className={`p-2 rounded-lg text-sm font-medium transition-colors ${
                        settings.fontSize === size
                          ? 'bg-red-600 text-white'
                          : 'bg-gray-50 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-600'
                      }`}
                    >
                      {size === 'small' ? 'Petit' : size === 'medium' ? 'Moyen' : 'Grand'}
                    </button>
                  ))}
                </div>
              </div>

              {/* Animations Toggle */}
              <div className="mb-4">
                <label className="flex items-center justify-between">
                  <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                    Animations
                  </span>
                  <button
                    onClick={toggleAnimations}
                    className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                      settings.animations ? 'bg-red-600' : 'bg-gray-200 dark:bg-gray-700'
                    }`}
                  >
                    <span
                      className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                        settings.animations ? 'translate-x-6' : 'translate-x-1'
                      }`}
                    />
                  </button>
                </label>
              </div>

              {/* Sounds Toggle */}
              <div className="mb-4">
                <label className="flex items-center justify-between">
                  <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                    Sons
                  </span>
                  <button
                    onClick={toggleSounds}
                    className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                      settings.sounds ? 'bg-red-600' : 'bg-gray-200 dark:bg-gray-700'
                    }`}
                  >
                    <span
                      className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                        settings.sounds ? 'translate-x-6' : 'translate-x-1'
                      }`}
                    />
                  </button>
                </label>
              </div>

              <button
                onClick={() => setShowSettings(false)}
                className="w-full mt-4 bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 py-2 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors"
              >
                Fermer
              </button>
            </div>
          </div>
        )}
      </header>
    </>
  );
};

export default ChatHeader;

