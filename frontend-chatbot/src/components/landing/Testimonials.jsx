import { Star, Quote } from 'lucide-react';

const Testimonials = () => {
  const testimonials = [
    {
      name: "Marie Dubois",
      role: "Responsable RH France",
      department: "Ressources Humaines",
      content: "Le chatbot a révolutionné notre façon de gérer les demandes RH. Nos employés obtiennent des réponses instantanées 24/7, ce qui améliore considérablement leur expérience. L'automatisation nous permet de nous concentrer sur des tâches à plus forte valeur ajoutée.",
      rating: 5,
      avatar: "MD"
    },
    {
      name: "Thomas Laurent",
      role: "Manager IT",
      department: "Systèmes d'Information",
      content: "L'intégration technique s'est déroulée sans accroc. La sécurité des données est exemplaire et la conformité RGPD est parfaitement respectée. L'interface est intuitive et l'adoption par les équipes a été immédiate.",
      rating: 5,
      avatar: "TL"
    },
    {
      name: "Sophie Martin",
      role: "Employée Marketing",
      department: "Marketing Digital",
      content: "Enfin un outil qui comprend vraiment nos questions ! Plus besoin d'attendre des heures pour connaître mon solde de congés ou les procédures internes. Le chatbot répond instantanément et avec précision. C'est un vrai gain de temps au quotidien.",
      rating: 5,
      avatar: "SM"
    },
    {
      name: "Jean-Pierre Rousseau",
      role: "Directeur Opérations",
      department: "Operations",
      content: "Les métriques parlent d'elles-mêmes : 85% de réduction du temps de traitement des demandes RH et 95% de satisfaction employé. Ce projet illustre parfaitement notre engagement dans la transformation digitale chez Nestlé.",
      rating: 5,
      avatar: "JPR"
    },
    {
      name: "Amélie Chen",
      role: "Nouvelle Employée",
      department: "Finance",
      content: "Mon onboarding a été facilité grâce au chatbot. Toutes mes questions de nouvelle arrivante ont trouvé des réponses claires et détaillées. L'accompagnement personnalisé m'a permis de m'intégrer rapidement dans l'équipe.",
      rating: 5,
      avatar: "AC"
    },
    {
      name: "David Wilson",
      role: "Chef de Projet",
      department: "Innovation",
      content: "La capacité d'apprentissage du chatbot est impressionnante. Il s'améliore continuellement et s'adapte aux spécificités de notre organisation. C'est exactement le type d'innovation dont Nestlé a besoin pour rester leader.",
      rating: 5,
      avatar: "DW"
    }
  ];

  const stats = [
    { value: "95%", label: "Satisfaction Employé" },
    { value: "85%", label: "Réduction Temps Traitement" },
    { value: "24/7", label: "Disponibilité" },
    { value: "2s", label: "Temps de Réponse Moyen" }
  ];

  return (
    <section id="testimonials" className="py-20 bg-gradient-to-br from-gray-50 to-blue-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="text-center mb-16">
          <h2 className="text-4xl font-bold text-gray-900 mb-4">
            Ce que disent nos <span className="text-red-600">Collaborateurs</span>
          </h2>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            Découvrez les retours d'expérience de nos employés et managers qui utilisent 
            quotidiennement notre assistant RH conversationnel.
          </p>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-8 mb-16">
          {stats.map((stat, index) => (
            <div key={index} className="text-center">
              <div className="bg-white rounded-2xl p-6 shadow-lg hover:shadow-xl transition-shadow duration-300">
                <div className="text-3xl lg:text-4xl font-bold text-red-600 mb-2">
                  {stat.value}
                </div>
                <div className="text-sm lg:text-base text-gray-600">
                  {stat.label}
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Testimonials Grid */}
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
          {testimonials.map((testimonial, index) => (
            <div key={index} className="bg-white rounded-2xl p-8 shadow-lg hover:shadow-xl transition-all duration-300 transform hover:-translate-y-1">
              {/* Quote Icon */}
              <div className="flex justify-between items-start mb-6">
                <Quote className="w-8 h-8 text-red-600 opacity-50" />
                <div className="flex">
                  {[...Array(testimonial.rating)].map((_, i) => (
                    <Star key={i} className="w-4 h-4 text-yellow-400 fill-current" />
                  ))}
                </div>
              </div>
              
              {/* Content */}
              <p className="text-gray-700 mb-6 leading-relaxed">
                "{testimonial.content}"
              </p>
              
              {/* Author */}
              <div className="flex items-center">
                <div className="w-12 h-12 bg-gradient-to-r from-red-500 to-red-600 rounded-full flex items-center justify-center text-white font-semibold mr-4">
                  {testimonial.avatar}
                </div>
                <div>
                  <div className="font-semibold text-gray-900">
                    {testimonial.name}
                  </div>
                  <div className="text-sm text-gray-600">
                    {testimonial.role}
                  </div>
                  <div className="text-xs text-red-600 font-medium">
                    {testimonial.department}
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Bottom CTA */}
        <div className="mt-16 text-center">
          <div className="bg-white rounded-2xl p-8 lg:p-12 shadow-lg">
            <h3 className="text-2xl lg:text-3xl font-bold text-gray-900 mb-4">
              Rejoignez les centaines d'employés satisfaits
            </h3>
            <p className="text-lg text-gray-600 mb-8 max-w-2xl mx-auto">
              Découvrez par vous-même pourquoi notre assistant RH conversationnel 
              transforme l'expérience employé chez Nestlé.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <button className="bg-red-600 hover:bg-red-700 text-white px-8 py-4 rounded-full font-semibold transition-all duration-300 transform hover:scale-105 hover:shadow-lg">
                Essayer Maintenant
              </button>
              {/* <button className="border-2 border-gray-300 text-gray-700 hover:border-red-600 hover:text-red-600 px-8 py-4 rounded-full font-semibold transition-all duration-300">
                Planifier une Démo
              </button> */}
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Testimonials;

