import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import TickerInput from './TickerInput';

describe('TickerInput', () => {
  it('renders input and button', () => {
    render(<TickerInput onFetch={() => {}} loading={false} />);
    expect(screen.getByPlaceholderText(/enter ticker/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /fetch data/i })).toBeInTheDocument();
  });

  it('calls onFetch with uppercase ticker on submit', () => {
    const onFetch = vi.fn();
    render(<TickerInput onFetch={onFetch} loading={false} />);

    const input = screen.getByPlaceholderText(/enter ticker/i);
    fireEvent.change(input, { target: { value: 'msft' } });
    fireEvent.click(screen.getByRole('button', { name: /fetch data/i }));

    expect(onFetch).toHaveBeenCalledWith('MSFT', '5y');
  });

  it('disables button when loading', () => {
    render(<TickerInput onFetch={() => {}} loading={true} />);
    expect(screen.getByRole('button')).toBeDisabled();
  });

  it('disables button when input is empty', () => {
    render(<TickerInput onFetch={() => {}} loading={false} />);
    const input = screen.getByPlaceholderText(/enter ticker/i);
    fireEvent.change(input, { target: { value: '' } });
    expect(screen.getByRole('button')).toBeDisabled();
  });
});
