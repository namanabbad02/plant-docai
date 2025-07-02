import React from 'react';
import { User, Bot } from 'lucide-react';
import { Message } from '../types';

interface ChatMessageProps {
  message: Message;
}

export const ChatMessage: React.FC<ChatMessageProps> = ({ message }) => {
  const isUser = message.role === 'user';

  return (
    <div className={`flex items-start gap-3 ${isUser ? 'flex-row-reverse' : ''}`}>
      <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${
        isUser ? 'bg-green-100' : 'bg-green-500'
      }`}>
        {isUser ? (
          <User className="w-5 h-5 text-green-600" />
        ) : (
          <Bot className="w-5 h-5 text-white" />
        )}
      </div>
      <div className={`flex-1 px-4 py-2 rounded-lg ${
        isUser ? 'bg-green-50 text-gray-800' : 'bg-white border text-gray-700'
      }`}>
        {message.image && (
          <div className="mb-2">
            <img src={message.image} alt="Uploaded plant" className="max-w-xs rounded-lg" />
          </div>
        )}
        <p className="whitespace-pre-wrap">{message.content}</p>
      </div>
    </div>
  );
};