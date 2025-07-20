import { 
  MessageCircle, 
  Brain, 
  Clock, 
  Shield, 
  Globe, 
  BarChart3, 
  Users, 
  Smartphone,
  FileText,
  Calendar,
  Search,
  Settings
} from 'lucide-react';

const Features = () => {
  const mainFeatures = [
    {
      icon: MessageCircle,
      title: "Interface Conversationnelle Intuitive",
      description: "Dialogue naturel en français avec compréhension contextuelle avancée",
      color: "from-blue-500 to-blue-600"
    },
    {
      icon: Brain,
      title: "Intelligence Artificielle Avancée",
      description: "Traitement du langage naturel et apprentissage automatique continu",
      color: "from-purple-500 to-purple-600"
    },
    {
      icon: Clock,
      title: "Disponibilité 24/7",
      description: "Support instantané à toute heure, tous les jours de l'année",
      color: "from-green-500 to-green-600"
    },
    {
      icon: Shield,
      title: "Sécurité Renforcée",
      description: "Conformité RGPD et chiffrement des données personnelles",
      color: "from-red-500 to-red-600"
    }
  ];

  const additionalFeatures = [
    {
      icon: Globe,
      title: "Support Multilingue",
      description: "Français, anglais et langues locales"
    },
    {
      icon: BarChart3,
      title: "Analytics Avancés",
      description: "Tableaux de bord et rapports détaillés"
    },
    {
      icon: Users,
      title: "Escalade Intelligente",
      description: "Transfert automatique vers les experts RH"
    },
    {
      icon: Smartphone,
      title: "Multi-plateforme",
      description: "Web, mobile et intégration Teams"
    },
    {
      icon: FileText,
      title: "Base de Connaissances",
      description: "Accès aux politiques et procédures"
    },
    {
      icon: Calendar,
      title: "Gestion des Congés",
      description: "Demandes et suivi automatisés"
    },
    {
      icon: Search,
      title: "Recherche Intelligente",
      description: "Trouvez rapidement l'information"
    },
    {
      icon: Settings,
      title: "Personnalisation",
      description: "Adaptation aux besoins spécifiques"
    }
  ];

  return (
    <section id="features" className="py-20 bg-white">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="text-center mb-16">
          <h2 className="text-4xl font-bold text-gray-900 mb-4">
            Fonctionnalités <span className="text-red-600">Avancées</span>
          </h2>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            Découvrez les capacités exceptionnelles de notre assistant RH conversationnel, 
            conçu pour répondre à tous vos besoins professionnels.
          </p>
        </div>

        {/* Main Features */}
        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8 mb-20">
          {mainFeatures.map((feature, index) => (
            <div key={index} className="group relative">
              <div className="bg-white rounded-2xl p-8 shadow-lg hover:shadow-2xl transition-all duration-300 transform hover:-translate-y-2 border border-gray-100">
                <div className={`w-16 h-16 bg-gradient-to-r ${feature.color} rounded-xl flex items-center justify-center mb-6 group-hover:scale-110 transition-transform duration-300`}>
                  <feature.icon className="w-8 h-8 text-white" />
                </div>
                <h3 className="text-xl font-semibold text-gray-900 mb-4">
                  {feature.title}
                </h3>
                <p className="text-gray-600">
                  {feature.description}
                </p>
              </div>
            </div>
          ))}
        </div>

        {/* Additional Features Grid */}
        <div className="bg-gray-50 rounded-3xl p-8 lg:p-12">
          <h3 className="text-3xl font-bold text-gray-900 text-center mb-12">
            Fonctionnalités Complètes
          </h3>
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            {additionalFeatures.map((feature, index) => (
              <div key={index} className="bg-white rounded-xl p-6 hover:shadow-lg transition-shadow duration-300 group">
                <div className="w-12 h-12 bg-red-100 rounded-lg flex items-center justify-center mb-4 group-hover:bg-red-600 transition-colors duration-300">
                  <feature.icon className="w-6 h-6 text-red-600 group-hover:text-white transition-colors duration-300" />
                </div>
                <h4 className="font-semibold text-gray-900 mb-2">
                  {feature.title}
                </h4>
                <p className="text-sm text-gray-600">
                  {feature.description}
                </p>
              </div>
            ))}
          </div>
        </div>

        {/* Use Cases */}
        <div className="mt-20">
          <h3 className="text-3xl font-bold text-gray-900 text-center mb-12">
            Cas d'Usage Principaux
          </h3>
          <div className="grid lg:grid-cols-3 gap-8">
            <div className="bg-gradient-to-br from-blue-50 to-blue-100 rounded-2xl p-8">
              <div className="w-14 h-14 bg-blue-600 rounded-xl flex items-center justify-center mb-6">
                <Users className="w-7 h-7 text-white" />
              </div>
              <h4 className="text-xl font-semibold text-gray-900 mb-4">
                Nouveaux Employés
              </h4>
              <p className="text-gray-700 mb-4">
                Accompagnement personnalisé pour l'onboarding et l'intégration des nouveaux collaborateurs.
              </p>
              <ul className="text-sm text-gray-600 space-y-2">
                <li>• Guide d'accueil interactif</li>
                <li>• Informations sur les équipes</li>
                <li>• Procédures administratives</li>
                <li>• Formation aux outils internes</li>
              </ul>
            </div>

            <div className="bg-gradient-to-br from-green-50 to-green-100 rounded-2xl p-8">
              <div className="w-14 h-14 bg-green-600 rounded-xl flex items-center justify-center mb-6">
                <Calendar className="w-7 h-7 text-white" />
              </div>
              <h4 className="text-xl font-semibold text-gray-900 mb-4">
                Gestion des Congés
              </h4>
              <p className="text-gray-700 mb-4">
                Simplification des demandes de congés et suivi en temps réel des soldes disponibles.
              </p>
              <ul className="text-sm text-gray-600 space-y-2">
                <li>• Demandes de congés instantanées</li>
                <li>• Vérification des soldes</li>
                <li>• Calendrier d'équipe</li>
                <li>• Notifications automatiques</li>
              </ul>
            </div>

            <div className="bg-gradient-to-br from-purple-50 to-purple-100 rounded-2xl p-8">
              <div className="w-14 h-14 bg-purple-600 rounded-xl flex items-center justify-center mb-6">
                <FileText className="w-7 h-7 text-white" />
              </div>
              <h4 className="text-xl font-semibold text-gray-900 mb-4">
                Support Quotidien
              </h4>
              <p className="text-gray-700 mb-4">
                Réponses instantanées aux questions fréquentes et accès aux ressources RH.
              </p>
              <ul className="text-sm text-gray-600 space-y-2">
                <li>• Politiques d'entreprise</li>
                <li>• Avantages sociaux</li>
                <li>• Contacts utiles</li>
                <li>• FAQ interactive</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Features;

