import React from 'react';
import './Header.css';

interface HeaderProps {
  title: string;
}

export class Header extends React.Component<HeaderProps> {
  render() {
    return (
      <header className="header">
        <div className="header-container">
          <h1 className="header-title">{this.props.title}</h1>
        </div>
      </header>
    );
  }
}
