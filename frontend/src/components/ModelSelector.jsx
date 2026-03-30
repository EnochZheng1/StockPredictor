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

export default function ModelSelector({ models, ensembleMethods = [], modelParams = {}, onRun, loading }) {
  const [selected, setSelected] = useState([]);
  const [selectedEnsembles, setSelectedEnsembles] = useState([]);
  const [steps, setSteps] = useState(30);
  const [paramOverrides, setParamOverrides] = useState({});
  const [expandedModel, setExpandedModel] = useState(null);

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

  const toggleExpand = (model) => {
    setExpandedModel(expandedModel === model ? null : model);
  };

  const setParam = (model, param, value) => {
    setParamOverrides((prev) => ({
      ...prev,
      [model]: { ...prev[model], [param]: value },
    }));
  };

  const getCleanParams = () => {
    // Only include params that differ from defaults
    const clean = {};
    for (const [model, params] of Object.entries(paramOverrides)) {
      if (!selected.includes(model)) continue;
      const defs = modelParams[model] || {};
      const overrides = {};
      for (const [key, value] of Object.entries(params)) {
        if (defs[key] && value !== defs[key].default) {
          overrides[key] = value;
        }
      }
      if (Object.keys(overrides).length > 0) clean[model] = overrides;
    }
    return clean;
  };

  const ensemblesDisabled = selected.length < 2;

  return (
    <div className="model-selector">
      <h3>Select Models</h3>
      <div className="model-list">
        <label className="model-checkboxes select-all">
          <input
            type="checkbox"
            checked={selected.length === models.length && models.length > 0}
            onChange={selectAll}
          />
          Select All
        </label>
        {models.map((model) => (
          <div key={model} className="model-item">
            <div className="model-item-row">
              <label>
                <input
                  type="checkbox"
                  checked={selected.includes(model)}
                  onChange={() => toggle(model)}
                />
                {MODEL_LABELS[model] || model}
              </label>
              {modelParams[model] && (
                <button
                  type="button"
                  className="tune-btn"
                  onClick={() => toggleExpand(model)}
                  title="Tune hyperparameters"
                >
                  {expandedModel === model ? "Hide" : "Tune"}
                </button>
              )}
            </div>
            {expandedModel === model && modelParams[model] && (
              <div className="param-panel">
                {Object.entries(modelParams[model]).map(([param, info]) => (
                  <div key={param} className="param-row">
                    <label title={info.description}>
                      {param}
                      <input
                        type="number"
                        value={paramOverrides[model]?.[param] ?? info.default}
                        onChange={(e) => {
                          const val = info.type === "float"
                            ? parseFloat(e.target.value)
                            : parseInt(e.target.value, 10);
                          if (!isNaN(val)) setParam(model, param, val);
                        }}
                        min={info.min}
                        max={info.max}
                        step={info.type === "float" ? 0.01 : 1}
                      />
                    </label>
                    <span className="param-desc">{info.description}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
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
        onClick={() => onRun(selected, steps, ensemblesDisabled ? [] : selectedEnsembles, getCleanParams())}
        disabled={loading || selected.length === 0}
      >
        {loading ? "Running Models..." : "Run Comparison"}
      </button>
    </div>
  );
}
