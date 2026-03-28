import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { useState } from "react";

export default function FeatureImportanceChart({ comparisonResults }) {
  const modelsWithImportance = (comparisonResults?.results || []).filter(
    (r) => r.feature_importance
  );

  const [selectedModel, setSelectedModel] = useState("");

  if (modelsWithImportance.length === 0) return null;

  const activeModel =
    modelsWithImportance.find((r) => r.model_name === selectedModel) ||
    modelsWithImportance[0];

  // Sort by importance descending, take top 15
  const chartData = Object.entries(activeModel.feature_importance)
    .map(([name, value]) => ({ name, importance: value }))
    .sort((a, b) => b.importance - a.importance)
    .slice(0, 15);

  return (
    <div className="feature-importance">
      <div className="feature-importance-header">
        <h3>Feature Importance</h3>
        {modelsWithImportance.length > 1 && (
          <select
            value={activeModel.model_name}
            onChange={(e) => setSelectedModel(e.target.value)}
          >
            {modelsWithImportance.map((r) => (
              <option key={r.model_name} value={r.model_name}>
                {r.model_name}
              </option>
            ))}
          </select>
        )}
      </div>
      <ResponsiveContainer width="100%" height={Math.max(300, chartData.length * 28)}>
        <BarChart data={chartData} layout="vertical" margin={{ left: 100 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis type="number" tick={{ fontSize: 11 }} />
          <YAxis
            type="category"
            dataKey="name"
            tick={{ fontSize: 11 }}
            width={95}
          />
          <Tooltip formatter={(v) => v.toFixed(4)} />
          <Bar dataKey="importance" fill="#2563eb" radius={[0, 4, 4, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
