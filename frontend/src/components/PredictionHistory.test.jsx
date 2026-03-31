import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import PredictionHistory from './PredictionHistory';

vi.mock('../api/stockApi', () => ({
  fetchHistory: vi.fn(),
}));

import { fetchHistory } from '../api/stockApi';

describe('PredictionHistory', () => {
  it('renders nothing without ticker', () => {
    const { container } = render(<PredictionHistory ticker="" />);
    expect(container.innerHTML).toBe('');
  });

  it('shows toggle button with history', async () => {
    fetchHistory.mockResolvedValue([
      { id: 1, model_name: 'RF', period: '5y', rmse: 1.5, mae: 1.2, r2: 0.9, created_at: '2024-01-01T10:00:00' },
    ]);
    render(<PredictionHistory ticker="AAPL" />);
    await waitFor(() => {
      expect(screen.getByText(/Prediction History/)).toBeInTheDocument();
    });
  });

  it('expands to show table', async () => {
    fetchHistory.mockResolvedValue([
      { id: 1, model_name: 'RF', period: '5y', rmse: 1.5, mae: 1.2, r2: 0.9, created_at: '2024-01-01T10:00:00' },
    ]);
    render(<PredictionHistory ticker="AAPL" />);
    await waitFor(() => screen.getByText(/Prediction History/));
    fireEvent.click(screen.getByText(/Show/));
    expect(screen.getByText('RF')).toBeInTheDocument();
  });
});
