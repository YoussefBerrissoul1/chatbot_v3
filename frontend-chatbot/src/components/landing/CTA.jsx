import { ArrowRight, MessageCircle, Play, Sparkles } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Link } from 'react-router-dom';

const CTA = () => {
  return (
    <section className="py-20 bg-gradient-to-r from-red-600 via-red-700 to-red-800 relative overflow-hidden">
      {/* Background Animation */}
      <div className="absolute inset-0">
        <div className="absolute top-0 left-0 w-full h-full bg-gradient-to-br from-red-600/20 to-transparent"></div>
        <div className="absolute -top-40 -right-40 w-80 h-80 bg-white/10 rounded-full blur-3xl"></div>
        <div className="absolute -bottom-40 -left-40 w-80 h-80 bg-white/10 rounded-full blur-3xl"></div>
      </div>

      <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center">
          {/* Badge */}
          <div className="inline-flex items-center px-4 py-2 bg-white/20 backdrop-blur-sm text-white rounded-full text-sm font-medium mb-8 animate-pulse">
            <Sparkles className="w-4 h-4 mr-2" />
            Prêt à révolutionner votre expérience RH ?
          </div>

          {/* Main Heading */}
          <h2 className="text-4xl sm:text-5xl lg:text-6xl font-bold text-white mb-6 leading-tight">
            Commencez votre
            <span className="block text-yellow-300">
              Transformation RH
            </span>
            <span className="block">
              dès Aujourd'hui
            </span>
          </h2>

          {/* Description */}
          <p className="text-xl text-red-100 mb-12 max-w-3xl mx-auto leading-relaxed">
            Rejoignez la révolution de l'assistance RH chez Nestlé. 
            Notre chatbot conversationnel vous attend pour simplifier votre quotidien professionnel 
            et vous offrir une expérience utilisateur exceptionnelle.
          </p>

          {/* CTA Buttons */}
          <div className="flex flex-col sm:flex-row gap-6 justify-center mb-16">
            <Link to="/chat">
              <Button className="bg-white text-red-600 hover:bg-gray-100 px-8 py-4 rounded-full text-lg font-semibold transition-all duration-300 transform hover:scale-105 hover:shadow-2xl">
                <MessageCircle className="w-5 h-5 mr-2" />
                Essayer le Chatbot
                <ArrowRight className="w-5 h-5 ml-2" />
              </Button>
            </Link>
            {/* <Button 
              variant="outline" 
              className="border-2 border-white text-white hover:bg-white hover:text-red-600 px-8 py-4 rounded-full text-lg font-semibold transition-all duration-300 transform hover:scale-105"
            >
              <Play className="w-5 h-5 mr-2" />
              Voir la Démo
            </Button> */}
          </div>

          {/* Features Grid */}
          <div className="grid md:grid-cols-3 gap-8 max-w-4xl mx-auto">
            <div className="text-center">
              <div className="w-16 h-16 bg-white/20 backdrop-blur-sm rounded-2xl flex items-center justify-center mx-auto mb-4">
                <MessageCircle className="w-8 h-8 text-white" />
              </div>
              <h3 className="text-xl font-semibold text-white mb-2">
                Interface Intuitive
              </h3>
              <p className="text-red-100">
                Conversation naturelle en français avec votre assistant RH personnel
              </p>
            </div>

            <div className="text-center">
              <div className="w-16 h-16 bg-white/20 backdrop-blur-sm rounded-2xl flex items-center justify-center mx-auto mb-4">
                <Sparkles className="w-8 h-8 text-white" />
              </div>
              <h3 className="text-xl font-semibold text-white mb-2">
                IA Avancée
              </h3>
              <p className="text-red-100">
                Intelligence artificielle qui apprend et s'améliore continuellement
              </p>
            </div>

            <div className="text-center">
              <div className="w-16 h-16 bg-white/20 backdrop-blur-sm rounded-2xl flex items-center justify-center mx-auto mb-4">
                <ArrowRight className="w-8 h-8 text-white" />
              </div>
              <h3 className="text-xl font-semibold text-white mb-2">
                Résultats Immédiats
              </h3>
              <p className="text-red-100">
                Réponses instantanées et solutions efficaces à vos questions RH
              </p>
            </div>
          </div>

          {/* Trust Indicators */}
          <div className="mt-16 pt-8 border-t border-white/20">
            <p className="text-red-100 mb-6">
              Déjà adopté par plus de 1000 employés Nestlé dans le monde
            </p>
            <div className="flex justify-center items-center space-x-8 opacity-70">
              <div className="text-white font-semibold">
                🔒 Sécurisé RGPD
              </div>
              <div className="text-white font-semibold">
                ⚡ Disponible 24/7
              </div>
              <div className="text-white font-semibold">
                🌍 Support Multilingue
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default CTA;

