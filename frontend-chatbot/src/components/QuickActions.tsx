import React from 'react';
import { User, Calendar, FileText, HelpCircle, Clock, Settings } from 'lucide-react';
import { ChatSettings } from '../types/chat';

interface QuickActionsProps {
  onActionClick: (action: string) => void;
  settings: ChatSettings;
}

const QuickActions: React.FC<QuickActionsProps> = ({ onActionClick, settings }) => {
  const actions = [
    { icon: Calendar, label: 'Congés', action: 'Consulter mes congés' },
    { icon: Clock, label: 'Horaires', action: 'Voir mes horaires' },
    { icon: HelpCircle, label: 'Aide', action: 'Aide et support' }
  ];

  return (
    <div className="px-6 py-4 bg-white dark:bg-gray-800 border-t border-gray-200 dark:border-gray-700">
      <div className="flex flex-wrap gap-2">
        {actions.map((action, index) => (
          <button
            key={index}
            onClick={() => onActionClick(action.action)}
            className={`
              flex items-center space-x-2 px-4 py-2 rounded-full text-sm font-medium
              bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300
              hover:bg-indigo-100 dark:hover:bg-indigo-900 hover:text-indigo-700 dark:hover:text-indigo-300
              transition-all duration-200 hover:scale-105 hover:shadow-md
              ${settings.animations ? 'animate-fade-in' : ''}
            `}
            style={{ animationDelay: `${index * 0.1}s` }}
          >
            <action.icon className="w-4 h-4" />
            <span>{action.label}</span>
          </button>
        ))}
      </div>
    </div>
  );
};

export default QuickActions;