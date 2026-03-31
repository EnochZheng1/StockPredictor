import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import ExportButton from './ExportButton';

const mockResults = {
  results: [
    {
      model_name: 'Linear Regression',
      test_predictions: [100, 101, 102],
      test_dates: ['2024-01-01', '2024-01-02', '2024-01-03'],
      future_predictions: [103, 104],
      future_dates: ['2024-01-04', '2024-01-05'],
    },
  ],
  summary: [
    { model_name: 'Linear Regression', rmse: 1.5, mae: 1.2, r2: 0.95 },
  ],
};

describe('ExportButton', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders nothing without results', () => {
    const { container } = render(<ExportButton comparisonResults={null} />);
    expect(container.innerHTML).toBe('');
  });

  it('renders two export buttons', () => {
    render(<ExportButton comparisonResults={mockResults} />);
    expect(screen.getByText('Export Metrics CSV')).toBeInTheDocument();
    expect(screen.getByText('Export Predictions CSV')).toBeInTheDocument();
  });

  it('creates blob on metrics export click', () => {
    render(<ExportButton comparisonResults={mockResults} />);
    const origCreate = document.createElement.bind(document);
    const mockClick = vi.fn();
    vi.spyOn(document, 'createElement').mockImplementation((tag) => {
      const el = origCreate(tag);
      if (tag === 'a') el.click = mockClick;
      return el;
    });
    fireEvent.click(screen.getByText('Export Metrics CSV'));
    expect(URL.createObjectURL).toHaveBeenCalled();
    document.createElement.mockRestore();
  });

  it('creates blob on predictions export click', () => {
    render(<ExportButton comparisonResults={mockResults} />);
    const origCreate = document.createElement.bind(document);
    const mockClick = vi.fn();
    vi.spyOn(document, 'createElement').mockImplementation((tag) => {
      const el = origCreate(tag);
      if (tag === 'a') el.click = mockClick;
      return el;
    });
    fireEvent.click(screen.getByText('Export Predictions CSV'));
    expect(URL.createObjectURL).toHaveBeenCalled();
    document.createElement.mockRestore();
  });
});
