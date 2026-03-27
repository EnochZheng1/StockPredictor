import axios from "axios";

const API = axios.create({
  baseURL: "http://localhost:4289/api",
});

export async function fetchStockData(ticker) {
  const { data } = await API.get(`/stocks/${ticker}`);
  return data;
}

export async function getAvailableModels() {
  const { data } = await API.get("/models");
  return data.models;
}

export async function runPrediction(ticker, modelName, steps = 30) {
  const { data } = await API.post("/predict", {
    ticker,
    model_name: modelName,
    steps,
  });
  return data;
}

export async function runComparison(ticker, modelNames, steps = 30) {
  const { data } = await API.post("/compare", {
    ticker,
    model_names: modelNames,
    steps,
  });
  return data;
}
