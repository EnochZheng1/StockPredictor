import axios from "axios";

const API = axios.create({
  baseURL: import.meta.env.VITE_API_URL || "http://localhost:4289/api",
  timeout: 300000, // 5 min timeout for model training
});

export async function fetchStockData(ticker) {
  const { data } = await API.get(`/stocks/${ticker}`);
  return data;
}

export async function getAvailableModels() {
  const { data } = await API.get("/models");
  return { models: data.models, ensembleMethods: data.ensemble_methods || [] };
}

export async function runPrediction(ticker, modelName, steps = 30) {
  const { data } = await API.post("/predict", {
    ticker,
    model_name: modelName,
    steps,
  });
  return data;
}

export async function runComparison(ticker, modelNames, steps = 30, ensembleMethods = []) {
  const { data } = await API.post("/compare", {
    ticker,
    model_names: modelNames,
    steps,
    ensemble_methods: ensembleMethods,
  });
  return data;
}
