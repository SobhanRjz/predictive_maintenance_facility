export type TabType = 'overview' | 'report-analysis' | 'ai-mode';

export interface Tab {
  id: TabType;
  label: string;
}

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}
