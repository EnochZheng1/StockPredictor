import '@testing-library/jest-dom';
import { vi } from 'vitest';

// Mock lightweight-charts
vi.mock('lightweight-charts', () => ({
  createChart: vi.fn(() => ({
    addCandlestickSeries: vi.fn(() => ({ setData: vi.fn(), priceScale: vi.fn(() => ({ applyOptions: vi.fn() })) })),
    addHistogramSeries: vi.fn(() => ({ setData: vi.fn(), priceScale: vi.fn(() => ({ applyOptions: vi.fn() })) })),
    addLineSeries: vi.fn(() => ({ setData: vi.fn() })),
    timeScale: vi.fn(() => ({ fitContent: vi.fn() })),
    applyOptions: vi.fn(),
    remove: vi.fn(),
  })),
}));

// Mock recharts
vi.mock('recharts', () => {
  const React = require('react');
  return {
    ResponsiveContainer: ({ children }) => React.createElement('div', { 'data-testid': 'responsive-container' }, children),
    LineChart: ({ children }) => React.createElement('div', { 'data-testid': 'line-chart' }, children),
    BarChart: ({ children }) => React.createElement('div', { 'data-testid': 'bar-chart' }, children),
    Line: () => null,
    Bar: () => null,
    XAxis: () => null,
    YAxis: () => null,
    CartesianGrid: () => null,
    Tooltip: () => null,
    Legend: () => null,
    ReferenceLine: () => null,
  };
});

// Mock WebSocket
class MockWebSocket {
  constructor(url) {
    this.url = url;
    this.readyState = 1;
    this.onmessage = null;
    this.onerror = null;
    this.onclose = null;
    MockWebSocket.instances.push(this);
  }
  send() {}
  close() {}
}
MockWebSocket.instances = [];
global.WebSocket = MockWebSocket;

// Mock URL methods
global.URL.createObjectURL = vi.fn(() => 'blob:mock-url');
global.URL.revokeObjectURL = vi.fn();
