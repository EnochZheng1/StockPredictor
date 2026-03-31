import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import LiveTicker from './LiveTicker';

describe('LiveTicker', () => {
  it('renders nothing without ticker', () => {
    const { container } = render(<LiveTicker ticker="" />);
    expect(container.innerHTML).toBe('');
  });

  it('renders nothing without data', () => {
    const { container } = render(<LiveTicker ticker="AAPL" />);
    // No WS data yet, should render nothing
    expect(container.innerHTML).toBe('');
  });

  it('creates WebSocket connection', () => {
    render(<LiveTicker ticker="AAPL" />);
    expect(WebSocket.instances.length).toBeGreaterThan(0);
  });
});
