import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import CandlestickChart from './CandlestickChart';

const mockStockData = {
  dates: ['2024-01-01', '2024-01-02', '2024-01-03'],
  open_prices: [100, 101, 102],
  high_prices: [102, 103, 104],
  low_prices: [99, 100, 101],
  close_prices: [101, 102, 103],
  volume: [1000000, 1100000, 1200000],
  indicators: { SMA_20: [100, 100.5, 101] },
};

describe('CandlestickChart', () => {
  it('renders nothing without data', () => {
    const { container } = render(<CandlestickChart stockData={null} />);
    expect(container.innerHTML).toBe('');
  });

  it('renders chart title with data', () => {
    render(<CandlestickChart stockData={mockStockData} />);
    expect(screen.getByText('Price Chart')).toBeInTheDocument();
  });

  it('renders overlay selector', () => {
    render(<CandlestickChart stockData={mockStockData} />);
    expect(screen.getByText('Volume')).toBeInTheDocument();
  });
});
