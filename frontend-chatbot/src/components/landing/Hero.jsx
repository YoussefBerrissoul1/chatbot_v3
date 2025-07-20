import { ArrowRight, Bot, Users, Zap } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Link } from 'react-router-dom';
import chatbotUI from '../../assets/images/chatbot-ui.png';

const Hero = () => {
  return (
    <section id="hero" className="relative min-h-screen flex items-center justify-center overflow-hidden bg-gradient-to-br from-red-50 via-white to-blue-50">
      {/* Background Animation */}
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute -top-40 -right-40 w-80 h-80 bg-red-100 rounded-full mix-blend-multiply filter blur-xl opacity-70 animate-blob"></div>
        <div className="absolute -bottom-40 -left-40 w-80 h-80 bg-blue-100 rounded-full mix-blend-multiply filter blur-xl opacity-70 animate-blob animation-delay-2000"></div>
        <div className="absolute top-40 left-40 w-80 h-80 bg-yellow-100 rounded-full mix-blend-multiply filter blur-xl opacity-70 animate-blob animation-delay-4000"></div>
      </div>

      <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20">
        <div className="grid lg:grid-cols-2 gap-12 items-center">
          {/* Content */}
          <div className="text-center lg:text-left">
            <div className="inline-flex items-center px-4 py-2 bg-red-100 text-red-800 rounded-full text-sm font-medium mb-6 animate-fade-in">
              <Bot className="w-4 h-4 mr-2" />
              Innovation RH Nestlé 2024
            </div>
            
            <h1 className="text-4xl sm:text-5xl lg:text-6xl font-bold text-gray-900 leading-tight mb-6 animate-slide-in-up">
              Assistant RH
              <span className="block text-red-600">
                Conversationnel
              </span>
              <span className="block text-gray-700 text-3xl sm:text-4xl lg:text-5xl">
                Nouvelle Génération
              </span>
            </h1>
            
            <p className="text-xl text-gray-600 mb-8 max-w-2xl animate-slide-in-up animation-delay-200">
              Découvrez notre solution innovante d'automatisation du support RH. 
              Un chatbot intelligent qui révolutionne l'expérience employé chez Nestlé 
              avec des réponses instantanées et personnalisées.
            </p>
            
            <div className="flex flex-col sm:flex-row gap-4 mb-12 animate-slide-in-up animation-delay-400">
              <Link to="/chat">
                <Button className="bg-red-600 hover:bg-red-700 text-white px-8 py-4 rounded-full text-lg font-semibold transition-all duration-300 transform hover:scale-105 hover:shadow-xl">
                  <MessageCircle className="w-5 h-5 mr-2" />
                  Essayer Maintenant
                  <ArrowRight className="w-5 h-5 ml-2" />
                </Button>
              </Link>
              <Button variant="outline" className="border-2 border-gray-300 text-gray-700 hover:border-red-600 hover:text-red-600 px-8 py-4 rounded-full text-lg font-semibold transition-all duration-300">
                Voir la Démo
              </Button>
            </div>
            
            {/* Stats */}
            <div className="grid grid-cols-3 gap-8 animate-slide-in-up animation-delay-600">
              <div className="text-center">
                <div className="flex items-center justify-center w-12 h-12 bg-red-100 rounded-full mx-auto mb-2">
                  <Users className="w-6 h-6 text-red-600" />
                </div>
                <div className="text-2xl font-bold text-gray-900">24/7</div>
                <div className="text-sm text-gray-600">Support Disponible</div>
              </div>
              <div className="text-center">
                <div className="flex items-center justify-center w-12 h-12 bg-blue-100 rounded-full mx-auto mb-2">
                  <Zap className="w-6 h-6 text-blue-600" />
                </div>
                <div className="text-2xl font-bold text-gray-900">&lt;2s</div>
                <div className="text-sm text-gray-600">Temps de Réponse</div>
              </div>
              <div className="text-center">
                <div className="flex items-center justify-center w-12 h-12 bg-green-100 rounded-full mx-auto mb-2">
                  <Bot className="w-6 h-6 text-green-600" />
                </div>
                <div className="text-2xl font-bold text-gray-900">95%</div>
                <div className="text-sm text-gray-600">Précision IA</div>
              </div>
            </div>
          </div>
          
          {/* Image */}
          <div className="relative animate-slide-in-up animation-delay-800">
            <div className="relative">
              <img 
                src={chatbotUI} 
                alt="Interface Chatbot Nestlé" 
                className="w-full max-w-lg mx-auto rounded-2xl shadow-2xl transform hover:scale-105 transition-transform duration-500"
              />
              <div className="absolute -top-4 -right-4 w-24 h-24 bg-red-500 rounded-full flex items-center justify-center animate-pulse">
                <Bot className="w-12 h-12 text-white" />
              </div>
              <div className="absolute -bottom-4 -left-4 w-16 h-16 bg-blue-500 rounded-full flex items-center justify-center animate-bounce">
                <Zap className="w-8 h-8 text-white" />
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Hero;

