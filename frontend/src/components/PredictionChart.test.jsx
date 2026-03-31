import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import PredictionChart from './PredictionChart';

const stockData = {
  dates: ['2024-01-01', '2024-01-02'],
  close_prices: [100, 101],
};

const comparisonResults = {
  results: [
    {
      model_name: 'Linear Regression',
      test_predictions: [100, 101],
      test_dates: ['2024-01-01', '2024-01-02'],
      future_predictions: [102, 103],
      future_dates: ['2024-01-03', '2024-01-04'],
    },
  ],
};

describe('PredictionChart', () => {
  it('renders nothing without data', () => {
    const { container } = render(<PredictionChart />);
    expect(container.innerHTML).toBe('');
  });

  it('renders with stock data only', () => {
    render(<PredictionChart stockData={stockData} />);
    expect(screen.getByText('Price Predictions')).toBeInTheDocument();
  });

  it('renders with comparison results', () => {
    render(<PredictionChart stockData={stockData} comparisonResults={comparisonResults} />);
    expect(screen.getByText('Price Predictions')).toBeInTheDocument();
  });

  it('renders chart container', () => {
    render(<PredictionChart stockData={stockData} comparisonResults={comparisonResults} />);
    expect(screen.getByTestId('responsive-container')).toBeInTheDocument();
  });
});
