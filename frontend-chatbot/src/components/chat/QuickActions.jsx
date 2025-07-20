import { 
  User, 
  Calendar, 
  FileText, 
  HelpCircle, 
  Clock, 
  Settings,
  Phone,
  MapPin,
  CreditCard,
  GraduationCap,
  Users,
  Home
} from 'lucide-react';

const QuickActions = ({ onActionClick, settings }) => {
  const actions = [
    { 
      icon: Calendar, 
      label: 'Congés', 
      action: 'Consulter mes congés et soldes disponibles',
      color: 'bg-blue-100 text-blue-600 hover:bg-blue-200'
    },
    { 
      icon: Clock, 
      label: 'Horaires', 
      action: 'Voir mes horaires de travail et planning',
      color: 'bg-green-100 text-green-600 hover:bg-green-200'
    },
    { 
      icon: CreditCard, 
      label: 'Paie', 
      action: 'Informations sur ma rémunération et avantages',
      color: 'bg-purple-100 text-purple-600 hover:bg-purple-200'
    },
    { 
      icon: GraduationCap, 
      label: 'Formation', 
      action: 'Catalogue de formations et développement professionnel',
      color: 'bg-orange-100 text-orange-600 hover:bg-orange-200'
    },
    { 
      icon: Users, 
      label: 'Équipe', 
      action: 'Contacts et informations sur mon équipe',
      color: 'bg-pink-100 text-pink-600 hover:bg-pink-200'
    },
    { 
      icon: Home, 
      label: 'Télétravail', 
      action: 'Politique et demandes de télétravail',
      color: 'bg-indigo-100 text-indigo-600 hover:bg-indigo-200'
    },
    { 
      icon: FileText, 
      label: 'Documents', 
      action: 'Accès aux documents RH et politiques',
      color: 'bg-gray-100 text-gray-600 hover:bg-gray-200'
    },
    { 
      icon: Phone, 
      label: 'Contact', 
      action: 'Contacts utiles et numéros d\'urgence',
      color: 'bg-red-100 text-red-600 hover:bg-red-200'
    },
    { 
      icon: HelpCircle, 
      label: 'Aide', 
      action: 'Aide et support technique',
      color: 'bg-yellow-100 text-yellow-600 hover:bg-yellow-200'
    }
  ];

  return (
    <div className="px-4 sm:px-6 py-4 bg-white dark:bg-gray-800 border-t border-gray-200 dark:border-gray-700">
      <div className="mb-3">
        <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300">
          Actions rapides
        </h3>
        <p className="text-xs text-gray-500 dark:text-gray-400">
          Cliquez sur une action pour commencer
        </p>
      </div>
      
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-3">
        {actions.map((action, index) => (
          <button
            key={index}
            onClick={() => onActionClick(action.action)}
            className={`
              flex flex-col items-center p-3 rounded-xl text-sm font-medium
              transition-all duration-200 hover:scale-105 hover:shadow-md
              ${action.color}
              ${settings.animations ? 'animate-fade-in' : ''}
            `}
            style={{ animationDelay: `${index * 0.05}s` }}
          >
            <action.icon className="w-6 h-6 mb-2" />
            <span className="text-center leading-tight">{action.label}</span>
          </button>
        ))}
      </div>

      {/* Popular Actions */}
      <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700">
        <p className="text-xs text-gray-500 dark:text-gray-400 mb-2">
          Actions populaires cette semaine
        </p>
        <div className="flex flex-wrap gap-2">
          <button
            onClick={() => onActionClick('Demande de congés pour les vacances d\'été')}
            className="px-3 py-1 bg-blue-50 dark:bg-blue-900/20 text-blue-700 dark:text-blue-300 rounded-full text-xs hover:bg-blue-100 dark:hover:bg-blue-900/40 transition-colors"
          >
            🏖️ Congés été
          </button>
          <button
            onClick={() => onActionClick('Politique de télétravail et jours autorisés')}
            className="px-3 py-1 bg-green-50 dark:bg-green-900/20 text-green-700 dark:text-green-300 rounded-full text-xs hover:bg-green-100 dark:hover:bg-green-900/40 transition-colors"
          >
            🏠 Télétravail
          </button>
          <button
            onClick={() => onActionClick('Formations disponibles en développement personnel')}
            className="px-3 py-1 bg-purple-50 dark:bg-purple-900/20 text-purple-700 dark:text-purple-300 rounded-full text-xs hover:bg-purple-100 dark:hover:bg-purple-900/40 transition-colors"
          >
            📚 Formations
          </button>
        </div>
      </div>
    </div>
  );
};

export default QuickActions;

