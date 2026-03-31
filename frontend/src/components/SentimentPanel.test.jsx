import { render, screen, waitFor } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import SentimentPanel from './SentimentPanel';

vi.mock('../api/stockApi', () => ({
  fetchSentiment: vi.fn(),
}));

import { fetchSentiment } from '../api/stockApi';

describe('SentimentPanel', () => {
  it('renders nothing without ticker', () => {
    const { container } = render(<SentimentPanel ticker="" />);
    expect(container.innerHTML).toBe('');
  });

  it('renders nothing while loading', () => {
    fetchSentiment.mockReturnValue(new Promise(() => {})); // never resolves
    const { container } = render(<SentimentPanel ticker="AAPL" />);
    expect(container.innerHTML).toBe('');
  });

  it('renders articles with data', async () => {
    fetchSentiment.mockResolvedValue({
      ticker: 'AAPL',
      articles: [
        { title: 'Good news', sentiment_label: 'positive', sentiment_score: 0.5, date: '2024-01-01', publisher: 'Test', url: '#' },
      ],
      avg_sentiment: 0.5,
      total_articles: 1,
    });
    render(<SentimentPanel ticker="AAPL" />);
    await waitFor(() => {
      expect(screen.getByText('Good news')).toBeInTheDocument();
    });
  });
});
