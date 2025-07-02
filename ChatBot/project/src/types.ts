export interface Message {
  id: string;
  content: string;
  role: 'user' | 'assistant';
  timestamp: Date;
  image?: string;
}

export interface ChatResponse {
  reply: string;
  confidence: number;
  disease?: string;
  treatment?: string;
}