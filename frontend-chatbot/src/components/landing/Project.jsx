import { Bot, Brain, Clock, Shield, Users, Zap } from 'lucide-react';
import nestleTeam from '../../assets/images/nestle-team.webp';

const Project = () => {
  const objectives = [
    {
      icon: Clock,
      title: "Réduction du Temps de Réponse",
      description: "Passer de plusieurs heures à moins de 2 secondes pour les demandes RH courantes"
    },
    {
      icon: Users,
      title: "Amélioration de l'Expérience Employé",
      description: "Support 24/7 disponible en français, anglais et autres langues locales"
    },
    {
      icon: Brain,
      title: "Intelligence Artificielle Avancée",
      description: "Traitement du langage naturel et apprentissage continu des interactions"
    },
    {
      icon: Shield,
      title: "Sécurité et Confidentialité",
      description: "Conformité RGPD et protection des données personnelles des employés"
    }
  ];

  const features = [
    "Gestion des congés et absences",
    "Informations sur les politiques RH",
    "Support pour les nouveaux employés",
    "Réponses aux questions fréquentes",
    "Escalade vers les équipes RH",
    "Intégration avec les systèmes existants"
  ];

  return (
    <section id="project" className="py-20 bg-gray-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="text-center mb-16">
          <div className="inline-flex items-center px-4 py-2 bg-blue-100 text-blue-800 rounded-full text-sm font-medium mb-6">
            <Bot className="w-4 h-4 mr-2" />
            Projet Innovation RH
          </div>
          <h2 className="text-4xl font-bold text-gray-900 mb-4">
            Développement d'un Chatbot Conversationnel
          </h2>
          <h3 className="text-2xl text-red-600 font-semibold mb-6">
            pour l'Automatisation du Support RH
          </h3>
          <p className="text-xl text-gray-600 max-w-4xl mx-auto">
            Un projet ambitieux visant à révolutionner l'expérience RH chez Nestlé grâce à 
            l'intelligence artificielle et aux technologies conversationnelles de pointe.
          </p>
        </div>

        <div className="grid lg:grid-cols-2 gap-16 items-center mb-16">
          {/* Content */}
          <div>
            <h3 className="text-3xl font-bold text-gray-900 mb-6">
              Vision du Projet
            </h3>
            <p className="text-lg text-gray-600 mb-6">
              Notre chatbot conversationnel représente l'avenir du support RH chez Nestlé. 
              Conçu pour comprendre et répondre aux besoins spécifiques de nos employés, 
              il combine intelligence artificielle avancée et expertise RH humaine.
            </p>
            <p className="text-lg text-gray-600 mb-8">
              Ce projet s'inscrit dans notre stratégie de transformation digitale, 
              visant à améliorer l'efficacité opérationnelle tout en renforçant 
              l'engagement et la satisfaction de nos collaborateurs.
            </p>
            
            {/* Key Features */}
            <div className="bg-white rounded-xl p-6 shadow-lg">
              <h4 className="text-xl font-semibold text-gray-900 mb-4">
                Fonctionnalités Clés
              </h4>
              <div className="grid grid-cols-2 gap-3">
                {features.map((feature, index) => (
                  <div key={index} className="flex items-center">
                    <div className="w-2 h-2 bg-red-600 rounded-full mr-3"></div>
                    <span className="text-sm text-gray-700">{feature}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
          
          {/* Image */}
          <div className="relative">
            <img 
              src={nestleTeam} 
              alt="Équipe RH Nestlé" 
              className="w-full rounded-2xl shadow-2xl"
            />
            <div className="absolute inset-0 bg-gradient-to-t from-black/30 to-transparent rounded-2xl"></div>
            <div className="absolute bottom-6 left-6 text-white">
              <p className="text-lg font-semibold">Équipe RH Nestlé</p>
              <p className="text-sm opacity-90">Innovation & Transformation Digitale</p>
            </div>
            
            {/* Floating Stats */}
            <div className="absolute -top-6 -right-6 bg-white rounded-xl p-4 shadow-xl">
              <div className="flex items-center">
                <div className="w-12 h-12 bg-green-100 rounded-full flex items-center justify-center mr-3">
                  <Zap className="w-6 h-6 text-green-600" />
                </div>
                <div>
                  <div className="text-2xl font-bold text-gray-900">85%</div>
                  <div className="text-xs text-gray-600">Automatisation</div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Objectives */}
        <div>
          <h3 className="text-3xl font-bold text-gray-900 text-center mb-12">
            Objectifs du Projet
          </h3>
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8">
            {objectives.map((objective, index) => (
              <div key={index} className="bg-white rounded-xl p-6 shadow-lg hover:shadow-xl transition-shadow duration-300 group">
                <div className="w-14 h-14 bg-gradient-to-br from-red-500 to-red-600 rounded-xl flex items-center justify-center mb-4 group-hover:scale-110 transition-transform duration-300">
                  <objective.icon className="w-7 h-7 text-white" />
                </div>
                <h4 className="text-xl font-semibold text-gray-900 mb-3">
                  {objective.title}
                </h4>
                <p className="text-gray-600">
                  {objective.description}
                </p>
              </div>
            ))}
          </div>
        </div>

        {/* Timeline */}
        <div className="mt-20 bg-white rounded-2xl p-8 lg:p-12 shadow-lg">
          <h3 className="text-3xl font-bold text-gray-900 text-center mb-12">
            Roadmap du Projet
          </h3>
          <div className="grid md:grid-cols-4 gap-8">
            <div className="text-center">
              <div className="w-16 h-16 bg-red-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <span className="text-2xl font-bold text-red-600">1</span>
              </div>
              <h4 className="font-semibold text-gray-900 mb-2">Analyse & Conception</h4>
              <p className="text-sm text-gray-600">Étude des besoins et architecture technique</p>
            </div>
            <div className="text-center">
              <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <span className="text-2xl font-bold text-blue-600">2</span>
              </div>
              <h4 className="font-semibold text-gray-900 mb-2">Développement</h4>
              <p className="text-sm text-gray-600">Création du chatbot et intégration IA</p>
            </div>
            <div className="text-center">
              <div className="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <span className="text-2xl font-bold text-green-600">3</span>
              </div>
              <h4 className="font-semibold text-gray-900 mb-2">Tests & Validation</h4>
              <p className="text-sm text-gray-600">Tests utilisateurs et optimisations</p>
            </div>
            <div className="text-center">
              <div className="w-16 h-16 bg-purple-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <span className="text-2xl font-bold text-purple-600">4</span>
              </div>
              <h4 className="font-semibold text-gray-900 mb-2">Déploiement</h4>
              <p className="text-sm text-gray-600">Mise en production et formation</p>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Project;

