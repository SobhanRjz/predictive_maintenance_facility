export interface IChatService {
  sendMessage(message: string): Promise<string>;
}

export class ChatService implements IChatService {
  private readonly apiKey: string;
  private readonly apiUrl: string = 'https://api.openai.com/v1/chat/completions';

  constructor(apiKey: string) {
    this.apiKey = apiKey;
  }

  async sendMessage(message: string): Promise<string> {
    const response = await fetch(this.apiUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${this.apiKey}`
      },
      body: JSON.stringify({
        model: 'gpt-3.5-turbo',
        messages: [
          { 
            role: 'system', 
            content: 'You are an AI assistant specialized in ML Gas & Oil analysis. You analyze sensor data and ML outputs to help explain system behavior and maintenance risks. Provide clear, technical explanations about anomalies, warnings, RUL estimation, and maintenance recommendations based on industrial sensor data.'
          },
          { role: 'user', content: message }
        ],
        temperature: 0.7
      })
    });

    if (!response.ok) {
      throw new Error(`API request failed: ${response.statusText}`);
    }

    const data = await response.json();
    return data.choices[0].message.content;
  }
}
