import React from 'react';

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true };
  }

  componentDidCatch(error, errorInfo) {
    // Log error to monitoring service if needed
    console.error('ErrorBoundary caught:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return <div style={{padding: 40, textAlign: 'center', color: 'red'}}>Une erreur est survenue. Veuillez recharger la page ou contacter le support.</div>;
    }
    return this.props.children;
  }
}

export default ErrorBoundary; 