import AnimatedHero from '../components/landing/AnimatedHero';
import About from '../components/landing/About';
import Project from '../components/landing/Project';
import Features from '../components/landing/Features';
import Testimonials from '../components/landing/Testimonials';
import CTA from '../components/landing/CTA';
import Header from '../components/landing/Header';
import Footer from '../components/landing/Footer';

const LandingPage = () => {
  return (
    <div className="min-h-screen">
      <Header />
      <AnimatedHero />
      <About />
      <Project />
      <Features />
      <Testimonials />
      <CTA />
      <Footer />
    </div>
  );
};

export default LandingPage;

