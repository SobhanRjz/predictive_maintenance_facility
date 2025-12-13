import React from 'react';
import { PrimaryButton } from '../components/PrimaryButton';
import './OverviewTab.css';

interface OverviewTabProps {}

export class OverviewTab extends React.Component<OverviewTabProps> {
  handleAction = () => {
    console.log('Overview action triggered');
  };

  render() {
    return (
      <div className="overview-tab">
        <div className="content-card">
          <h2 className="section-title">Overview</h2>
          <p className="section-description">
            Monitor and analyze system performance metrics
          </p>
          <div className="action-container">
            <PrimaryButton
              label="View Details"
              onClick={this.handleAction}
            />
          </div>
        </div>
      </div>
    );
  }
}
