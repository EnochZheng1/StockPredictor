import { useState } from "react";

const MODEL_LABELS = {
  arima: "ARIMA",
  linear_regression: "Linear Regression",
  lstm: "LSTM",
  random_forest: "Random Forest",
  xgboost: "XGBoost",
  prophet: "Prophet",
  polynomial_regression: "Polynomial Regression",
};

export default function ModelSelector({ models, onRun, loading }) {
  const [selected, setSelected] = useState([]);
  const [steps, setSteps] = useState(30);

  const toggle = (model) => {
    setSelected((prev) =>
      prev.includes(model) ? prev.filter((m) => m !== model) : [...prev, model]
    );
  };

  const selectAll = () => {
    setSelected(selected.length === models.length ? [] : [...models]);
  };

  return (
    <div className="model-selector">
      <h3>Select Models</h3>
      <div className="model-checkboxes">
        <label className="select-all">
          <input
            type="checkbox"
            checked={selected.length === models.length}
            onChange={selectAll}
          />
          Select All
        </label>
        {models.map((model) => (
          <label key={model}>
            <input
              type="checkbox"
              checked={selected.includes(model)}
              onChange={() => toggle(model)}
            />
            {MODEL_LABELS[model] || model}
          </label>
        ))}
      </div>
      <div className="steps-input">
        <label>
          Forecast days:
          <input
            type="number"
            value={steps}
            onChange={(e) => setSteps(Number(e.target.value))}
            min={1}
            max={365}
          />
        </label>
      </div>
      <button
        onClick={() => onRun(selected, steps)}
        disabled={loading || selected.length === 0}
      >
        {loading ? "Running Models..." : "Run Comparison"}
      </button>
    </div>
  );
}
