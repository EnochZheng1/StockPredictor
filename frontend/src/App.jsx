import { useState, useEffect } from "react";
import Dashboard from "./pages/Dashboard";
import "./App.css";

function App() {
  const [dark, setDark] = useState(() => localStorage.getItem("theme") === "dark");

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", dark ? "dark" : "light");
    localStorage.setItem("theme", dark ? "dark" : "light");
  }, [dark]);

  return (
    <div className="app">
      <header className="app-header">
        <div>
          <h1>StockPredictor</h1>
          <p>Multi-model stock price prediction & comparison</p>
        </div>
        <button className="theme-toggle" onClick={() => setDark(!dark)}>
          {dark ? "Light Mode" : "Dark Mode"}
        </button>
      </header>
      <main>
        <Dashboard />
      </main>
    </div>
  );
}

export default App;
