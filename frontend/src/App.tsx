import React from 'react';
import { Header } from './components/Header';
import { TabNavigation } from './components/TabNavigation';
import { OverviewTab } from './pages/OverviewTab';
import { ReportAnalysisTab } from './pages/ReportAnalysisTab';
import { AIModeTab } from './pages/AIModeTab';
import { Tab, TabType } from './types';
import './styles/variables.css';
import './styles/reset.css';
import './App.css';

interface AppState {
  activeTab: TabType;
  isDarkMode: boolean;
}

export class App extends React.Component<{}, AppState> {
  private readonly tabs: Tab[] = [
    { id: 'overview', label: 'Overview' },
    { id: 'report-analysis', label: 'Report Analysis' },
    { id: 'ai-mode', label: 'AI Mode' }
  ];

  constructor(props: {}) {
    super(props);
    this.state = {
      activeTab: 'overview',
      isDarkMode: true
    };
  }

  componentDidMount(): void {
    this.applyTheme();
  }

  componentDidUpdate(_: {}, prevState: AppState): void {
    if (prevState.isDarkMode !== this.state.isDarkMode) {
      this.applyTheme();
    }
  }

  private applyTheme(): void {
    document.documentElement.setAttribute('data-theme', this.state.isDarkMode ? 'dark' : 'light');
  }

  handleTabChange = (tabId: TabType): void => {
    this.setState({ activeTab: tabId });
  };

  handleThemeToggle = (): void => {
    this.setState(prev => ({ isDarkMode: !prev.isDarkMode }));
  };

  render() {
    const { activeTab, isDarkMode } = this.state;

    return (
      <div className="app">
        <Header title="ML Gas Oil Analysis" />
        <TabNavigation
          tabs={this.tabs}
          activeTab={activeTab}
          onTabChange={this.handleTabChange}
          isDarkMode={isDarkMode}
          onThemeToggle={this.handleThemeToggle}
        />
        <main className="main-content">
          <div style={{ display: activeTab === 'overview' ? 'block' : 'none' }}>
            <OverviewTab />
          </div>
          <div style={{ display: activeTab === 'report-analysis' ? 'block' : 'none' }}>
            <ReportAnalysisTab isDarkMode={isDarkMode} />
          </div>
          <div style={{ display: activeTab === 'ai-mode' ? 'block' : 'none' }}>
            <AIModeTab isDarkMode={isDarkMode} />
          </div>
        </main>
      </div>
    );
  }
}
