import { Mail, Phone, MapPin, Globe, Linkedin, Twitter, Facebook, MessageCircle } from 'lucide-react';
import { Link } from 'react-router-dom';
import nestleLogo from '../../assets/images/nestle-logo.jpg';

const Footer = () => {
  const currentYear = new Date().getFullYear();

  const footerLinks = {
    product: [
      { name: 'Fonctionnalités', href: '#features' },
      { name: 'Sécurité', href: '#security' },
      { name: 'Intégrations', href: '#integrations' },
      { name: 'API', href: '#api' }
    ],
    company: [
      { name: 'À propos de Nestlé', href: '#about' },
      { name: 'Carrières', href: 'https://www.nestlejobs.com' },
      { name: 'Presse', href: 'https://www.nestle.com/media' },
      { name: 'Investisseurs', href: 'https://www.nestle.com/investors' }
    ],
    support: [
      { name: 'Centre d\'aide', href: '#help' },
      { name: 'Documentation', href: '#docs' },
      { name: 'Formation', href: '#training' },
      { name: 'Contact', href: '#contact' }
    ],
    legal: [
      { name: 'Politique de confidentialité', href: '#privacy' },
      { name: 'Conditions d\'utilisation', href: '#terms' },
      { name: 'RGPD', href: '#gdpr' },
      { name: 'Cookies', href: '#cookies' }
    ]
  };

  const socialLinks = [
    { name: 'LinkedIn', icon: Linkedin, href: 'https://www.linkedin.com/company/nestle' },
    { name: 'Twitter', icon: Twitter, href: 'https://twitter.com/nestle' },
    { name: 'Facebook', icon: Facebook, href: 'https://www.facebook.com/Nestle' }
  ];

  const contactInfo = [
    { icon: Mail, text: 'rh-support@nestle.com', href: 'mailto:rh-support@nestle.com' },
    { icon: Phone, text: '+33 1 23 45 67 89', href: 'tel:+33123456789' },
    { icon: MapPin, text: 'Noisiel, France', href: '#' }
  ];

  return (
    <footer id="contact" className="bg-gray-900 text-white">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Main Footer Content */}
        <div className="py-16">
          <div className="grid lg:grid-cols-6 gap-8">
            {/* Company Info */}
            <div className="lg:col-span-2">
              <div className="flex items-center space-x-3 mb-6">
                <img 
                  src={nestleLogo} 
                  alt="Nestlé" 
                  className="h-8 w-auto object-contain filter brightness-0 invert"
                />
                <div>
                  <h3 className="text-xl font-bold">Assistant RH Nestlé</h3>
                  <p className="text-sm text-gray-400">Innovation & Transformation</p>
                </div>
              </div>
              <p className="text-gray-400 mb-6 leading-relaxed">
                Révolutionnez votre expérience RH avec notre assistant conversationnel intelligent. 
                Disponible 24/7 pour répondre à toutes vos questions professionnelles.
              </p>
              
              {/* Contact Info */}
              <div className="space-y-3">
                {contactInfo.map((contact, index) => (
                  <a 
                    key={index}
                    href={contact.href}
                    className="flex items-center text-gray-400 hover:text-white transition-colors duration-200"
                  >
                    <contact.icon className="w-4 h-4 mr-3" />
                    {contact.text}
                  </a>
                ))}
              </div>
            </div>

            {/* Product Links */}
            <div>
              <h4 className="text-lg font-semibold mb-6">Produit</h4>
              <ul className="space-y-3">
                {footerLinks.product.map((link, index) => (
                  <li key={index}>
                    <a 
                      href={link.href}
                      className="text-gray-400 hover:text-white transition-colors duration-200"
                    >
                      {link.name}
                    </a>
                  </li>
                ))}
              </ul>
            </div>

            {/* Company Links */}
            <div>
              <h4 className="text-lg font-semibold mb-6">Entreprise</h4>
              <ul className="space-y-3">
                {footerLinks.company.map((link, index) => (
                  <li key={index}>
                    <a 
                      href={link.href}
                      className="text-gray-400 hover:text-white transition-colors duration-200"
                      target={link.href.startsWith('http') ? '_blank' : '_self'}
                      rel={link.href.startsWith('http') ? 'noopener noreferrer' : ''}
                    >
                      {link.name}
                    </a>
                  </li>
                ))}
              </ul>
            </div>

            {/* Support Links */}
            <div>
              <h4 className="text-lg font-semibold mb-6">Support</h4>
              <ul className="space-y-3">
                {footerLinks.support.map((link, index) => (
                  <li key={index}>
                    <a 
                      href={link.href}
                      className="text-gray-400 hover:text-white transition-colors duration-200"
                    >
                      {link.name}
                    </a>
                  </li>
                ))}
              </ul>
            </div>

            {/* Legal Links */}
            <div>
              <h4 className="text-lg font-semibold mb-6">Légal</h4>
              <ul className="space-y-3">
                {footerLinks.legal.map((link, index) => (
                  <li key={index}>
                    <a 
                      href={link.href}
                      className="text-gray-400 hover:text-white transition-colors duration-200"
                    >
                      {link.name}
                    </a>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </div>

        {/* Newsletter Section */}
        <div className="py-8 border-t border-gray-800">
          <div className="grid md:grid-cols-2 gap-8 items-center">
            <div>
              <h4 className="text-xl font-semibold mb-2">
                Restez informé des nouveautés
              </h4>
              <p className="text-gray-400">
                Recevez les dernières mises à jour sur notre assistant RH et les innovations Nestlé.
              </p>
            </div>
            <div className="flex flex-col sm:flex-row gap-4">
              <input
                type="email"
                placeholder="Votre adresse email"
                className="flex-1 px-4 py-3 bg-gray-800 border border-gray-700 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-red-600 transition-colors duration-200"
              />
              <button className="bg-red-600 hover:bg-red-700 text-white px-6 py-3 rounded-lg font-semibold transition-colors duration-200 whitespace-nowrap">
                S'abonner
              </button>
            </div>
          </div>
        </div>

        {/* Bottom Footer */}
        <div className="py-8 border-t border-gray-800">
          <div className="flex flex-col md:flex-row justify-between items-center">
            <div className="flex flex-col md:flex-row items-center space-y-4 md:space-y-0 md:space-x-8">
              <p className="text-gray-400">
                © {currentYear} Nestlé S.A. Tous droits réservés.
              </p>
              <div className="flex items-center space-x-4">
                <Globe className="w-4 h-4 text-gray-400" />
                <select className="bg-transparent text-gray-400 border-none focus:outline-none">
                  <option value="fr">Français</option>
                  <option value="en">English</option>
                  <option value="de">Deutsch</option>
                </select>
              </div>
            </div>

            <div className="flex items-center space-x-6 mt-4 md:mt-0">
              {/* Social Links */}
              <div className="flex space-x-4">
                {socialLinks.map((social, index) => (
                  <a
                    key={index}
                    href={social.href}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="w-10 h-10 bg-gray-800 rounded-full flex items-center justify-center text-gray-400 hover:text-white hover:bg-red-600 transition-all duration-200"
                  >
                    <social.icon className="w-5 h-5" />
                  </a>
                ))}
              </div>

              {/* Quick Access to Chatbot */}
              <Link to="/chat">
                <button className="flex items-center space-x-2 bg-red-600 hover:bg-red-700 text-white px-4 py-2 rounded-full transition-colors duration-200">
                  <MessageCircle className="w-4 h-4" />
                  <span className="text-sm font-medium">Chatbot</span>
                </button>
              </Link>
            </div>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;

