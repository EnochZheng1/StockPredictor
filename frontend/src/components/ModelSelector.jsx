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

const ENSEMBLE_LABELS = {
  ensemble_average: "Simple Average",
  ensemble_weighted: "Weighted Average (by RMSE)",
  ensemble_stacking: "Stacking (Ridge)",
};

export default function ModelSelector({ models, ensembleMethods = [], onRun, loading }) {
  const [selected, setSelected] = useState([]);
  const [selectedEnsembles, setSelectedEnsembles] = useState([]);
  const [steps, setSteps] = useState(30);

  const toggle = (model) => {
    setSelected((prev) =>
      prev.includes(model) ? prev.filter((m) => m !== model) : [...prev, model]
    );
  };

  const toggleEnsemble = (method) => {
    setSelectedEnsembles((prev) =>
      prev.includes(method) ? prev.filter((m) => m !== method) : [...prev, method]
    );
  };

  const selectAll = () => {
    setSelected(selected.length === models.length ? [] : [...models]);
  };

  const ensemblesDisabled = selected.length < 2;

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

      {ensembleMethods.length > 0 && (
        <div className="ensemble-section">
          <h4>Ensemble Methods</h4>
          {ensemblesDisabled && (
            <p className="ensemble-disabled-note">Select at least 2 models to enable ensembles</p>
          )}
          <div className="model-checkboxes">
            {ensembleMethods.map((method) => (
              <label key={method} className={ensemblesDisabled ? "disabled" : ""}>
                <input
                  type="checkbox"
                  checked={selectedEnsembles.includes(method)}
                  onChange={() => toggleEnsemble(method)}
                  disabled={ensemblesDisabled}
                />
                {ENSEMBLE_LABELS[method] || method}
              </label>
            ))}
          </div>
        </div>
      )}

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
        onClick={() => onRun(selected, steps, ensemblesDisabled ? [] : selectedEnsembles)}
        disabled={loading || selected.length === 0}
      >
        {loading ? "Running Models..." : "Run Comparison"}
      </button>
    </div>
  );
}
