const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
};

const processMessage = (message: string) => {
  const currentTime = new Date().toLocaleTimeString();
  const lowerMessage = message.toLowerCase();

  // Greetings
  if (/^(hi|hello|hey|greetings|howdy)/.test(lowerMessage)) {
    return {
      reply: "Hello! I'm here to help you with plant-related questions. How can I assist you today?",
      confidence: 1.0
    };
  }

  // Time queries
  if (lowerMessage.includes('time')) {
    return {
      reply: `The current time is ${currentTime}. I'm available 24/7 to help you with plant disease detection and advice!`,
      confidence: 1.0
    };
  }

  // Bot introduction/identity
  if (lowerMessage.includes('who are you') || lowerMessage.includes('what are you') || lowerMessage.includes('tell me about you') || lowerMessage.includes('about yourself')) {
    return {
      reply: "I'm PlantDoc AI, a specialized artificial intelligence designed to help identify plant diseases and provide care recommendations. I can analyze plant images and answer questions about plant health. While I'm knowledgeable about plants, I work best when you show me images of plants you're concerned about.",
      confidence: 1.0
    };
  }

  // Creator/origin
  if (lowerMessage.includes('who made you') || lowerMessage.includes('who created you')) {
    return {
      reply: "I was created by a team of developers and plant experts to help people identify and treat plant diseases. I'm constantly learning and improving to provide better assistance!",
      confidence: 1.0
    };
  }

  // Name questions
  if (lowerMessage.includes('your name') || lowerMessage.includes("what's your name")) {
    return {
      reply: "I'm PlantDoc AI, your plant health assistant. I specialize in identifying plant diseases and providing care recommendations.",
      confidence: 1.0
    };
  }

  // Default responses for plant-related queries
  if (lowerMessage.includes('plant') || lowerMessage.includes('disease') || lowerMessage.includes('leaf')) {
    return {
      reply: "I'd be happy to help you with your plant-related question. For the most accurate diagnosis, please share a clear image of your plant, focusing on any areas of concern.",
      confidence: 0.8
    };
  }

  // Default response for unrelated topics
  return {
    reply: "I specialize in plant disease detection and care advice. While I can chat briefly, I'm most helpful when discussing plants and analyzing plant images. Would you like to know more about how I can help with your plants?",
    confidence: 0.6
  };
};

Deno.serve(async (req) => {
  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers: corsHeaders });
  }

  try {
    const { message } = await req.json();
    const response = processMessage(message);
    
    return new Response(
      JSON.stringify(response),
      { 
        headers: { 
          ...corsHeaders,
          'Content-Type': 'application/json'
        } 
      }
    );
  } catch (error) {
    return new Response(
      JSON.stringify({ error: error.message }),
      { 
        status: 500,
        headers: { 
          ...corsHeaders,
          'Content-Type': 'application/json'
        }
      }
    );
  }
});