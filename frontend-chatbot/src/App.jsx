import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Suspense, lazy } from 'react';
import './App.css';

const LandingPage = lazy(() => import('./pages/LandingPage'));
const ChatPage = lazy(() => import('./pages/ChatPage'));

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-background">
        <Suspense fallback={<div>Chargement...</div>}>
          <Routes>
            <Route path="/" element={<LandingPage />} />
            <Route path="/chat" element={<ChatPage />} />
          </Routes>
        </Suspense>
      </div>
    </Router>
  );
}

export default App;

