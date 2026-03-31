import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import FeatureImportanceChart from './FeatureImportanceChart';

const resultsWithImportance = {
  results: [
    {
      model_name: 'Random Forest',
      feature_importance: { SMA_20: 0.3, RSI: 0.2, MACD: 0.15, EMA_20: 0.1, Momentum: 0.05 },
    },
  ],
};

const resultsWithout = {
  results: [
    { model_name: 'ARIMA', feature_importance: null },
  ],
};

describe('FeatureImportanceChart', () => {
  it('renders nothing without importance data', () => {
    const { container } = render(<FeatureImportanceChart comparisonResults={resultsWithout} />);
    expect(container.innerHTML).toBe('');
  });

  it('renders chart heading with importance data', () => {
    render(<FeatureImportanceChart comparisonResults={resultsWithImportance} />);
    expect(screen.getByText('Feature Importance')).toBeInTheDocument();
  });

  it('renders nothing with null results', () => {
    const { container } = render(<FeatureImportanceChart comparisonResults={null} />);
    expect(container.innerHTML).toBe('');
  });
});
