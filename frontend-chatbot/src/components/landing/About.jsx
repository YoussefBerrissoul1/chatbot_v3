import { Award, Globe, Heart, Leaf } from 'lucide-react';
import nestleOffice from '../../assets/images/nestle-office.jpg';

const About = () => {
  const values = [
    {
      icon: Heart,
      title: "Passion",
      description: "Nous mettons notre passion au service de l'alimentation et des boissons pour améliorer la qualité de vie."
    },
    {
      icon: Globe,
      title: "Excellence",
      description: "Nous nous efforçons d'atteindre l'excellence dans tout ce que nous faisons, partout dans le monde."
    },
    {
      icon: Leaf,
      title: "Respect",
      description: "Nous respectons l'avenir en agissant de manière responsable aujourd'hui."
    },
    {
      icon: Award,
      title: "Innovation",
      description: "Nous innovons constamment pour créer de la valeur partagée pour nos parties prenantes."
    }
  ];

  const stats = [
    { number: "152", label: "Années d'Histoire", suffix: "" },
    { number: "186", label: "Pays de Présence", suffix: "" },
    { number: "273", label: "Milliards CHF", suffix: "de Chiffre d'Affaires" },
    { number: "270", label: "Mille Employés", suffix: "dans le Monde" }
  ];

  return (
    <section id="about" className="py-20 bg-white">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="text-center mb-16">
          <h2 className="text-4xl font-bold text-gray-900 mb-4">
            À Propos de <span className="text-red-600">Nestlé</span>
          </h2>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            Leader mondial de l'alimentation et des boissons, Nestlé s'engage depuis plus de 150 ans 
            à libérer le pouvoir de l'alimentation pour améliorer la qualité de vie, aujourd'hui et pour les générations futures.
          </p>
        </div>

        <div className="grid lg:grid-cols-2 gap-16 items-center mb-20">
          {/* Content */}
          <div>
            <h3 className="text-3xl font-bold text-gray-900 mb-6">
              "Good food, Good life"
            </h3>
            <p className="text-lg text-gray-600 mb-6">
              Chez Nestlé, nous croyons que l'alimentation a le pouvoir d'améliorer la qualité de vie. 
              Notre mission est de libérer ce pouvoir pour tous, partout dans le monde.
            </p>
            <p className="text-lg text-gray-600 mb-8">
              Avec une présence dans 186 pays et plus de 270 000 employés, nous nous engageons 
              à créer de la valeur partagée en contribuant à des communautés plus saines et à un avenir plus durable.
            </p>
            
            {/* Values */}
            <div className="grid grid-cols-2 gap-6">
              {values.map((value, index) => (
                <div key={index} className="group">
                  <div className="flex items-center mb-3">
                    <div className="w-10 h-10 bg-red-100 rounded-lg flex items-center justify-center mr-3 group-hover:bg-red-600 transition-colors duration-300">
                      <value.icon className="w-5 h-5 text-red-600 group-hover:text-white transition-colors duration-300" />
                    </div>
                    <h4 className="font-semibold text-gray-900">{value.title}</h4>
                  </div>
                  <p className="text-sm text-gray-600">{value.description}</p>
                </div>
              ))}
            </div>
          </div>
          
          {/* Image */}
          <div className="relative">
            <img 
              src={nestleOffice} 
              alt="Bureau Nestlé" 
              className="w-full rounded-2xl shadow-2xl"
            />
            <div className="absolute inset-0 bg-gradient-to-t from-black/20 to-transparent rounded-2xl"></div>
            <div className="absolute bottom-6 left-6 text-white">
              <p className="text-sm font-medium">Siège Social Nestlé</p>
              <p className="text-xs opacity-90">Casablanca, Maroc</p>
            </div>
          </div>
        </div>

        {/* Stats */}
        <div className="bg-gradient-to-r from-red-600 to-red-700 rounded-2xl p-8 lg:p-12">
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-8">
            {stats.map((stat, index) => (
              <div key={index} className="text-center text-white">
                <div className="text-3xl lg:text-4xl font-bold mb-2">
                  {stat.number}
                </div>
                <div className="text-sm lg:text-base opacity-90">
                  {stat.label}
                </div>
                {stat.suffix && (
                  <div className="text-xs opacity-75 mt-1">
                    {stat.suffix}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
};

export default About;

