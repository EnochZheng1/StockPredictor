import Dashboard from "./pages/Dashboard";
import "./App.css";

function App() {
  return (
    <div className="app">
      <header className="app-header">
        <h1>StockPredictor</h1>
        <p>Multi-model stock price prediction & comparison</p>
      </header>
      <main>
        <Dashboard />
      </main>
    </div>
  );
}

export default App;
