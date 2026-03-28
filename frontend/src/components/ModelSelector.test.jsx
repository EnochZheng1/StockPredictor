import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import ModelSelector from './ModelSelector';

const MODELS = ['linear_regression', 'random_forest', 'xgboost'];
const ENSEMBLE_METHODS = ['ensemble_average', 'ensemble_weighted'];

describe('ModelSelector', () => {
  it('renders model checkboxes', () => {
    render(<ModelSelector models={MODELS} onRun={() => {}} loading={false} />);
    expect(screen.getByText('Linear Regression')).toBeInTheDocument();
    expect(screen.getByText('Random Forest')).toBeInTheDocument();
    expect(screen.getByText('XGBoost')).toBeInTheDocument();
  });

  it('disables run button when no models selected', () => {
    render(<ModelSelector models={MODELS} onRun={() => {}} loading={false} />);
    expect(screen.getByRole('button', { name: /run comparison/i })).toBeDisabled();
  });

  it('enables run button when a model is selected', () => {
    render(<ModelSelector models={MODELS} onRun={() => {}} loading={false} />);
    fireEvent.click(screen.getByText('Linear Regression'));
    expect(screen.getByRole('button', { name: /run comparison/i })).toBeEnabled();
  });

  it('select all toggles all models', () => {
    render(<ModelSelector models={MODELS} onRun={() => {}} loading={false} />);
    fireEvent.click(screen.getByText('Select All'));
    const checkboxes = screen.getAllByRole('checkbox');
    // All should be checked (Select All + 3 models)
    checkboxes.forEach((cb) => expect(cb).toBeChecked());
  });

  it('disables ensemble checkboxes when fewer than 2 models selected', () => {
    render(
      <ModelSelector
        models={MODELS}
        ensembleMethods={ENSEMBLE_METHODS}
        onRun={() => {}}
        loading={false}
      />
    );
    expect(screen.getByText(/select at least 2 models/i)).toBeInTheDocument();
  });

  it('enables ensemble checkboxes when 2+ models selected', () => {
    render(
      <ModelSelector
        models={MODELS}
        ensembleMethods={ENSEMBLE_METHODS}
        onRun={() => {}}
        loading={false}
      />
    );
    fireEvent.click(screen.getByText('Linear Regression'));
    fireEvent.click(screen.getByText('Random Forest'));
    expect(screen.queryByText(/select at least 2 models/i)).not.toBeInTheDocument();
  });
});
