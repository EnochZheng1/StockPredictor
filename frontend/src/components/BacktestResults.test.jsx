import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import BacktestResults from './BacktestResults';

const mockData = {
  buy_hold_return: 15.5,
  best_strategy: 'Random Forest',
  results: [
    {
      model_name: 'Random Forest',
      total_return: 20.5,
      buy_hold_return: 15.5,
      sharpe_ratio: 1.2,
      max_drawdown: 5.3,
      win_rate: 55.0,
      num_trades: 100,
      equity_curve: [1.0, 1.05, 1.1, 1.15, 1.2],
      dates: ['d1', 'd2', 'd3', 'd4', 'd5'],
    },
    {
      model_name: 'Linear Regression',
      total_return: -3.2,
      buy_hold_return: 15.5,
      sharpe_ratio: -0.5,
      max_drawdown: 12.1,
      win_rate: 40.0,
      num_trades: 80,
      equity_curve: [1.0, 0.98, 0.96, 0.95, 0.97],
      dates: ['d1', 'd2', 'd3', 'd4', 'd5'],
    },
  ],
};

describe('BacktestResults', () => {
  it('renders nothing without data', () => {
    const { container } = render(<BacktestResults backtestData={null} />);
    expect(container.innerHTML).toBe('');
  });

  it('renders model names in table', () => {
    render(<BacktestResults backtestData={mockData} />);
    expect(screen.getAllByText('Random Forest').length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText('Linear Regression')).toBeInTheDocument();
  });

  it('shows buy hold return', () => {
    render(<BacktestResults backtestData={mockData} />);
    expect(screen.getByText(/15\.5%/)).toBeInTheDocument();
  });

  it('shows best strategy badge', () => {
    render(<BacktestResults backtestData={mockData} />);
    expect(screen.getByText('Best')).toBeInTheDocument();
  });

  it('applies positive/negative styling', () => {
    render(<BacktestResults backtestData={mockData} />);
    expect(screen.getByText('20.5%')).toHaveClass('positive');
    expect(screen.getByText('-3.2%')).toHaveClass('negative');
  });
});
