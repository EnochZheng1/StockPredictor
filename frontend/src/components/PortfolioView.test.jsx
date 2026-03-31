import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import PortfolioView from './PortfolioView';

vi.mock('../api/stockApi', () => ({
  runPortfolio: vi.fn(),
}));

import { runPortfolio } from '../api/stockApi';

const MODELS = ['linear_regression', 'random_forest'];

describe('PortfolioView', () => {
  it('renders form elements', () => {
    render(<PortfolioView models={MODELS} />);
    expect(screen.getByPlaceholderText(/AAPL/)).toBeInTheDocument();
    expect(screen.getByText('Compare')).toBeInTheDocument();
  });

  it('shows model options in select', () => {
    render(<PortfolioView models={MODELS} />);
    expect(screen.getByText('Linear Regression')).toBeInTheDocument();
    expect(screen.getByText('Random Forest')).toBeInTheDocument();
  });

  it('shows results table on success', async () => {
    runPortfolio.mockResolvedValue({
      model_name: 'Random Forest',
      results: [
        { ticker: 'AAPL', current_price: 150, predicted_price: 155, predicted_change: 3.33, rmse: 1.5, r2: 0.9 },
      ],
    });
    render(<PortfolioView models={MODELS} />);
    fireEvent.click(screen.getByText('Compare'));
    await waitFor(() => {
      expect(screen.getByText('AAPL')).toBeInTheDocument();
    });
  });

  it('shows error on failure', async () => {
    runPortfolio.mockRejectedValue({ response: { data: { detail: 'Error occurred' } } });
    render(<PortfolioView models={MODELS} />);
    fireEvent.click(screen.getByText('Compare'));
    await waitFor(() => {
      expect(screen.getByText('Error occurred')).toBeInTheDocument();
    });
  });
});
