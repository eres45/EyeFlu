import React from 'react';
import EyeFluDetection from './components/EyeFluDetection';

function App() {
  return (
    <div className="App min-h-screen bg-gray-100">
      <div className="container mx-auto px-4 py-8">
        <header className="text-center mb-8">
          <h1 className="text-4xl font-bold text-blue-600">Eye Flu Detection System</h1>
          <p className="text-gray-600 mt-2">AI-Powered Eye Health Screening</p>
        </header>

        <main>
          <EyeFluDetection />
        </main>

        <footer className="text-center mt-8 text-gray-500">
          <p>© 2024 Eye Flu Detection. For medical advice, consult a healthcare professional.</p>
        </footer>
      </div>
    </div>
  );
}

export default App;