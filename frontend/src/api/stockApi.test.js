import { describe, it, expect, vi, beforeEach } from 'vitest';

const { mockGet, mockPost } = vi.hoisted(() => {
  const mockGet = vi.fn();
  const mockPost = vi.fn();
  return { mockGet, mockPost };
});

vi.mock('axios', () => ({
  default: {
    create: () => ({
      get: mockGet,
      post: mockPost,
    }),
  },
}));

import { fetchStockData, getAvailableModels, runComparison, runBacktest, runPortfolio, fetchSentiment } from './stockApi';

beforeEach(() => {
  mockGet.mockReset();
  mockPost.mockReset();
});

describe('stockApi', () => {
  it('fetchStockData calls correct endpoint', async () => {
    mockGet.mockResolvedValue({ data: { ticker: 'AAPL' } });
    await fetchStockData('AAPL', '5y');
    expect(mockGet).toHaveBeenCalledWith('/stocks/AAPL', { params: { period: '5y' } });
  });

  it('getAvailableModels transforms response', async () => {
    mockGet.mockResolvedValue({ data: { models: ['lr'], ensemble_methods: ['avg'], model_params: {} } });
    const result = await getAvailableModels();
    expect(result.models).toEqual(['lr']);
    expect(result.ensembleMethods).toEqual(['avg']);
  });

  it('runComparison sends correct body', async () => {
    mockPost.mockResolvedValue({ data: {} });
    await runComparison('AAPL', ['lr'], 30, '5y', ['avg'], {});
    expect(mockPost).toHaveBeenCalledWith('/compare', {
      ticker: 'AAPL', model_names: ['lr'], steps: 30, period: '5y',
      ensemble_methods: ['avg'], model_params: {},
    });
  });

  it('runBacktest sends correct body', async () => {
    mockPost.mockResolvedValue({ data: {} });
    await runBacktest('AAPL', ['lr'], '5y', {});
    expect(mockPost).toHaveBeenCalledWith('/backtest', {
      ticker: 'AAPL', model_names: ['lr'], period: '5y', model_params: {},
    });
  });

  it('runPortfolio sends correct body', async () => {
    mockPost.mockResolvedValue({ data: {} });
    await runPortfolio(['AAPL', 'MSFT'], 'lr', 30, '5y');
    expect(mockPost).toHaveBeenCalledWith('/portfolio', {
      tickers: ['AAPL', 'MSFT'], model_name: 'lr', steps: 30, period: '5y',
    });
  });

  it('fetchSentiment calls correct endpoint', async () => {
    mockGet.mockResolvedValue({ data: { articles: [] } });
    await fetchSentiment('AAPL');
    expect(mockGet).toHaveBeenCalledWith('/sentiment/AAPL');
  });
});
