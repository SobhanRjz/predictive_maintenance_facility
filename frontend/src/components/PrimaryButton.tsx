import React from 'react';
import './PrimaryButton.css';

interface PrimaryButtonProps {
  label: string;
  onClick: () => void;
  disabled?: boolean;
}

export class PrimaryButton extends React.Component<PrimaryButtonProps> {
  render() {
    const { label, onClick, disabled = false } = this.props;

    return (
      <button
        className="primary-button"
        onClick={onClick}
        disabled={disabled}
      >
        {label}
      </button>
    );
  }
}
