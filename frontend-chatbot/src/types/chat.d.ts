export interface Message {
  id: string;
  content: string;
  sender: 'user' | 'bot';
  timestamp: Date;
  type: 'text' | 'image' | 'file';
  status?: 'sent' | 'delivered' | 'read';
}

export interface ChatSettings {
  theme: 'light' | 'dark';
  fontSize: 'small' | 'medium' | 'large';
  animations: boolean;
  sounds: boolean;
}

