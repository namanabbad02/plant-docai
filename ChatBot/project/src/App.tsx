import React, { useState, useRef, useEffect } from 'react';
import { Send, Leaf, Upload } from 'lucide-react';
import { useDropzone } from 'react-dropzone';
import { ChatMessage } from './components/ChatMessage';
import { Message, ChatResponse } from './types';

function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    setMessages([{
      id: 'welcome',
      content: "Hello! I'm PlantDoc AI. Upload an image of your plant, and I'll help identify potential diseases and suggest solutions. You can also ask me questions about plant diseases.",
      role: 'assistant',
      timestamp: new Date(),
    }]);
  }, []);

  const onDrop = async (acceptedFiles: File[]) => {
    if (acceptedFiles.length === 0 || isLoading) return;

    const file = acceptedFiles[0];
    if (!file.type.startsWith('image/')) {
      alert('Please upload an image file');
      return;
    }

    setIsLoading(true);

    try {
      const reader = new FileReader();
      reader.onload = async () => {
        const base64Image = (reader.result as string).split(',')[1];
        
        const userMessage: Message = {
          id: Date.now().toString(),
          content: 'Please analyze this plant image',
          role: 'user',
          timestamp: new Date(),
          image: URL.createObjectURL(file)
        };

        setMessages(prev => [...prev, userMessage]);

        try {
          const response = await fetch(`${import.meta.env.VITE_SUPABASE_URL}/functions/v1/chat`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
              'Authorization': `Bearer ${import.meta.env.VITE_SUPABASE_ANON_KEY}`,
            },
            body: JSON.stringify({ image: base64Image }),
          });

          const data: ChatResponse = await response.json();

          const assistantMessage: Message = {
            id: (Date.now() + 1).toString(),
            content: `${data.reply}\n\nConfidence: ${(data.confidence * 100).toFixed(1)}%\n\nRecommended Treatment: ${data.treatment}`,
            role: 'assistant',
            timestamp: new Date(),
          };

          setMessages(prev => [...prev, assistantMessage]);
        } catch (error) {
          console.error('Error:', error);
        }
      };
      reader.readAsDataURL(file);
    } catch (error) {
      console.error('Error:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png']
    },
    disabled: isLoading
  });

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      content: input.trim(),
      role: 'user',
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await fetch(`${import.meta.env.VITE_SUPABASE_URL}/functions/v1/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${import.meta.env.VITE_SUPABASE_ANON_KEY}`,
        },
        body: JSON.stringify({ message: input.trim() }),
      });

      const data: ChatResponse = await response.json();

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: data.reply,
        role: 'assistant',
        timestamp: new Date(),
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Error:', error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-white">
      <div className="max-w-7xl mx-auto px-4 py-6">
        {/* Header */}
        <div className="flex items-center justify-center gap-3 mb-8 border-b pb-4">
          <Leaf className="w-8 h-8 text-green-600" />
          <h1 className="text-3xl font-bold text-gray-800">PlantDoc AI</h1>
        </div>

        <div className="grid grid-cols-12 gap-6">
          {/* Left Column - Upload & Instructions */}
          <div className="col-span-12 md:col-span-4 lg:col-span-3 space-y-4">
            <div className="bg-white rounded-lg shadow-sm border p-4">
              <h2 className="text-lg font-semibold text-gray-800 mb-2">Upload Plant Image</h2>
              <p className="text-sm text-gray-600 mb-3">Select an image for disease analysis.</p>
              
              <div
                {...getRootProps()}
                className={`border-2 border-dashed rounded-lg p-6 text-center cursor-pointer transition-colors ${
                  isDragActive ? 'border-green-500 bg-green-50' : 'border-gray-300 hover:border-green-400'
                } ${isLoading ? 'opacity-50 cursor-not-allowed' : ''}`}
              >
                <input {...getInputProps()} />
                <Upload className="w-6 h-6 mx-auto mb-2 text-gray-400" />
                <p className="text-sm text-gray-600">
                  {isDragActive ? "Drop the image here" : "Click or drag to upload"}
                </p>
              </div>
            </div>

            <div className="bg-green-50 rounded-lg p-4">
              <h3 className="text-md font-semibold text-gray-800 mb-3">How it works</h3>
              <ol className="space-y-2 text-sm text-gray-700">
                <li className="flex gap-2">
                  <span className="font-medium">1.</span>
                  Upload a clear image of the affected plant part.
                </li>
                <li className="flex gap-2">
                  <span className="font-medium">2.</span>
                  Our AI will analyze the image for potential diseases.
                </li>
                <li className="flex gap-2">
                  <span className="font-medium">3.</span>
                  Ask questions about the diagnosis or plant care.
                </li>
              </ol>
            </div>
          </div>

          {/* Right Column - Chat Section */}
          <div className="col-span-12 md:col-span-8 lg:col-span-9">
            <div className="bg-white rounded-lg shadow-sm border flex flex-col h-[700px]">
              <div className="flex-1 overflow-y-auto p-6 space-y-4">
                {messages.map(message => (
                  <ChatMessage key={message.id} message={message} />
                ))}
                <div ref={messagesEndRef} />
              </div>

              <form onSubmit={handleSubmit} className="p-4 border-t">
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    placeholder="Ask about plant diseases or type your message..."
                    className="flex-1 px-4 py-2 rounded-full border border-gray-300 focus:outline-none focus:ring-2 focus:ring-green-500"
                    disabled={isLoading}
                  />
                  <button
                    type="submit"
                    disabled={isLoading}
                    className="px-4 py-2 bg-green-600 text-white rounded-full hover:bg-green-700 transition-colors disabled:bg-gray-400 disabled:cursor-not-allowed"
                  >
                    <Send className="w-5 h-5" />
                  </button>
                </div>
              </form>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;