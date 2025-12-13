import React from 'react';
import { Tab, TabType } from '../types';
import './TabNavigation.css';

interface TabNavigationProps {
  tabs: Tab[];
  activeTab: TabType;
  onTabChange: (tabId: TabType) => void;
  isDarkMode: boolean;
  onThemeToggle: () => void;
}

export class TabNavigation extends React.Component<TabNavigationProps> {
  render() {
    const { tabs, activeTab, onTabChange, isDarkMode, onThemeToggle } = this.props;

    return (
      <nav className="tab-navigation">
        <div className="tab-container">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              className={`tab-button ${activeTab === tab.id ? 'active' : ''}`}
              onClick={() => onTabChange(tab.id)}
            >
              {tab.label}
            </button>
          ))}
        </div>
        <button className="theme-toggle" onClick={onThemeToggle} title={isDarkMode ? 'Light Mode' : 'Dark Mode'}>
          {isDarkMode ? '☀️' : '🌙'}
        </button>
      </nav>
    );
  }
}
