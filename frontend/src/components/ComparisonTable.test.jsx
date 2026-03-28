import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import ComparisonTable from './ComparisonTable';

const SUMMARY = [
  { model_name: 'Random Forest', rmse: 1.5, mae: 1.2, r2: 0.95 },
  { model_name: 'Linear Regression', rmse: 3.0, mae: 2.5, r2: 0.85 },
  { model_name: 'Ensemble (Average)', rmse: 2.0, mae: 1.8, r2: 0.90 },
];

describe('ComparisonTable', () => {
  it('renders nothing when summary is empty', () => {
    const { container } = render(<ComparisonTable summary={[]} bestModel="" />);
    expect(container.innerHTML).toBe('');
  });

  it('renders all model rows', () => {
    render(<ComparisonTable summary={SUMMARY} bestModel="Random Forest" />);
    expect(screen.getByText('Random Forest')).toBeInTheDocument();
    expect(screen.getByText('Linear Regression')).toBeInTheDocument();
    expect(screen.getByText('Ensemble (Average)')).toBeInTheDocument();
  });

  it('shows Best badge on best model', () => {
    render(<ComparisonTable summary={SUMMARY} bestModel="Random Forest" />);
    expect(screen.getByText('Best')).toBeInTheDocument();
  });

  it('formats metrics to 4 decimal places', () => {
    render(<ComparisonTable summary={SUMMARY} bestModel="Random Forest" />);
    expect(screen.getByText('1.5000')).toBeInTheDocument();
    expect(screen.getByText('0.9500')).toBeInTheDocument();
  });
});
