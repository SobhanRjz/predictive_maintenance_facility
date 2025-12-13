import React from 'react';
import './ReportAnalysisTab.css';

interface GrafanaPanel {
  id: string;
  title: string;
}

interface TimeRange {
  from: string;
  to: string;
}

interface Props {
  isDarkMode: boolean;
}

interface State {
  timeRange: TimeRange;
  customFrom: string;
  customTo: string;
  iframeKeys: Map<string, number>;
}

class GrafanaIframe extends React.PureComponent<{ url: string; title: string }> {
  render() {
    return (
      <iframe
        src={this.props.url}
        className="grafana-iframe"
        frameBorder="0"
        title={this.props.title}
        loading="eager"
      />
    );
  }
}

export class ReportAnalysisTab extends React.Component<Props, State> {
  private static readonly BASE_URL = 'http://localhost:3000/d-solo/adkjmmt/csv-generator';
  private urlCache: Map<string, string> = new Map();
  
  private static readonly PANELS: GrafanaPanel[] = [
    { id: 'panel-2', title: 'accelerometer_g' },
    { id: 'panel-14', title: 'power_consumption_kw' },
    { id: 'panel-15', title: 'sound_intensity db' },
    { id: 'panel-1', title: 'bearing_temp_c' },
    { id: 'panel-13', title: 'supply_voltage_v' },
    { id: 'panel-3', title: 'vibration_velocity : mm/s' },
    { id: 'panel-12', title: 'motor_current_a' },
    { id: 'panel-5', title: 'oil_temp_c' },
    { id: 'panel-11', title: 'flow_rate_m3_h' },
    { id: 'panel-4', title: 'shaft_displacement_um' },
    { id: 'panel-10', title: 'outlet_pressure_bar' },
    { id: 'panel-6', title: 'casing_temp_c' },
    { id: 'panel-9', title: 'inlet_pressure_bar' },
    { id: 'panel-7', title: 'inlet_fluid_temp_c' },
    { id: 'panel-8', title: 'outlet_fluid_temp_c' }
  ];

  constructor(props: Props) {
    super(props);
    this.state = {
      timeRange: { from: 'now-5m', to: 'now' },
      customFrom: '',
      customTo: '',
      iframeKeys: new Map()
    };
  }

  componentDidUpdate(prevProps: Props): void {
    if (prevProps.isDarkMode !== this.props.isDarkMode) {
      this.urlCache.clear();
      this.setState(prev => ({
        iframeKeys: new Map(Array.from(prev.iframeKeys).map(([k, v]) => [k, v + 1]))
      }));
    }
  }

  private buildPanelUrl(panelId: string): string {
    const { from, to } = this.state.timeRange;
    const theme = this.props.isDarkMode ? 'dark' : 'light';
    const cacheKey = `${panelId}-${from}-${to}-${theme}`;
    
    if (this.urlCache.has(cacheKey)) {
      return this.urlCache.get(cacheKey)!;
    }
    
    const params = new URLSearchParams({
      orgId: '1',
      theme,
      timezone: 'browser',
      refresh: 'auto',
      panelId,
      from,
      to,
      __feature: 'dashboardSceneSolo=true'
    });
    
    const url = `${ReportAnalysisTab.BASE_URL}?${params}`;
    this.urlCache.set(cacheKey, url);
    return url;
  }

  private setTimeRange = (from: string, to: string): void => {
    this.setState(prevState => ({
      timeRange: { from, to },
      iframeKeys: new Map(Array.from(prevState.iframeKeys).map(([k, v]) => [k, v + 1]))
    }));
  };

  private applyCustomRange = (): void => {
    const { customFrom, customTo } = this.state;
    if (customFrom && customTo) {
      const from = new Date(customFrom).getTime().toString();
      const to = new Date(customTo).getTime().toString();
      this.setTimeRange(from, to);
    }
  };

  render() {
    return (
      <div className="report-analysis-tab">
        <div className="content-header">
          <h2 className="section-title">Report Analysis</h2>
          <p className="section-description">Real-time analytics and system monitoring dashboards</p>

          <div className="time-range-selector">
            <button onClick={() => this.setTimeRange('now-5m', 'now')}>Last 5min</button>
            <button onClick={() => this.setTimeRange('now-10m', 'now')}>Last 10min</button>
            <button onClick={() => this.setTimeRange('now-30m', 'now')}>Last 30min</button>
            <button onClick={() => this.setTimeRange('now-1h', 'now')}>Last 1h</button>

            <div className="custom-range">
              <input
                type="datetime-local"
                value={this.state.customFrom}
                onChange={(e) => this.setState({ customFrom: e.target.value })}
              />
              <span>to</span>
              <input
                type="datetime-local"
                value={this.state.customTo}
                onChange={(e) => this.setState({ customTo: e.target.value })}
              />
              <button onClick={this.applyCustomRange}>Apply</button>
            </div>
          </div>
        </div>

        <div className="top-status-panel">
          <div className="status-header">
            <h3 className="status-title">System Status Overview</h3>
          </div>
          <div className="status-content">
            <iframe
              src={this.buildPanelUrl('panel-16')}
              width="100%"
              height="250"
              frameBorder="0"
              title="System Status"
              loading="eager"
            />
          </div>
        </div>

        <div className="grafana-grid">
          {ReportAnalysisTab.PANELS.map((panel) => {
            const iframeKey = this.state.iframeKeys.get(panel.id) || 0;
            return (
              <div key={panel.id} className="grafana-panel-card">
                <div className="panel-header">
                  <h3 className="panel-title">{panel.title}</h3>
                </div>
                <div className="panel-content">
                  <GrafanaIframe 
                    key={`${panel.id}-${iframeKey}`}
                    url={this.buildPanelUrl(panel.id)}
                    title={panel.title}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>
    );
  }
}
