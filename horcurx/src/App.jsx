import React, { useState, useRef, useEffect, useCallback } from 'react';
import { 
  ArrowUp, MessageSquare, AlertTriangle, Shield, TrendingUp, 
  DollarSign, BarChart2, Brain, Newspaper, ThumbsUp, ThumbsDown, 
  Send, Search, X, Check, Globe 
} from 'lucide-react';

// --- API Endpoints ---
const CHAT_API_ENDPOINT = 'http://localhost:8000/api/chat';
const FEEDBACK_API_ENDPOINT = 'http://localhost:8000/api/feedback'; // <-- NEW

const INITIAL_MESSAGE = {
  id: 'initial',
  type: 'bot',
  text: "Welcome to the Dealership AI Strategy Agent. I can provide real-time pricing, demand forecasts, and market insights. How can I assist you with inventory management today?",
  report_data: null,
  chart_data: null,
  feedback_pending: null
};

// --- 1. Main Chatbot Component ---
export default function App() {
  const [messages, setMessages] = useState([INITIAL_MESSAGE]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const endOfMessagesRef = useRef(null);

  // --- Feedback State ---
  // Stores the message ID and original query that we are collecting feedback for
  const [feedbackContext, setFeedbackContext] = useState(null); // { messageId: '...', originalQuery: '...' }
  const [feedbackInput, setFeedbackInput] = useState('');

  // --- Scroll to bottom on new message ---
  useEffect(() => {
    endOfMessagesRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // --- Handle Main Message Sending ---
  const handleSendMessage = useCallback(async (e) => {
    if (e) e.preventDefault();
    if (isLoading || !input.trim()) return;

    const userMessage = {
      id: `user-${Date.now()}`,
      type: 'user',
      text: input,
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);
    setFeedbackContext(null); // Clear any pending feedback when a new query is sent

    try {
      const response = await fetch(CHAT_API_ENDPOINT, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: userMessage.text }),
      });

      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);

      const data = await response.json();

      const botMessage = {
        id: `bot-${Date.now()}`,
        type: 'bot',
        text: data.response_text,
        report_data: data.report_data,
        chart_data: data.chart_data,
        // Check if this is a discount report to enable feedback buttons
        feedback_pending: data.report_data?.intent === 'get_discount' ? userMessage.text : null
      };
      setMessages(prev => [...prev, botMessage]);

    } catch (error) {
      console.error('Failed to send message:', error);
      const errorMessage = {
        id: `err-${Date.now()}`,
        type: 'bot',
        text: "Sorry, I'm having trouble connecting to my services. Please check the API server and try again.",
        isError: true,
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  }, [input, isLoading]);


  // --- Handle Feedback Submission ---
  const handleFeedbackSubmit = useCallback(async (e) => {
    e.preventDefault();
    if (!feedbackInput.trim() || !feedbackContext) return;

    const { messageId, originalQuery } = feedbackContext;

    try {
      // Send feedback to the new endpoint
      await fetch(FEEDBACK_API_ENDPOINT, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: originalQuery,
          feedback_reason: feedbackInput,
        }),
      });

      // Update the original bot message to remove feedback UI
      setMessages(prev =>
        prev.map(msg =>
          msg.id === messageId
            ? { ...msg, feedback_pending: null, text: `${msg.text}\n\n**[Feedback Logged]** Thank you. Your correction has been saved.` }
            : msg
        )
      );

      // Add a confirmation message
      const confirmationMessage = {
        id: `bot-fb-${Date.now()}`,
        type: 'bot',
        text: "Thank you, your feedback has been logged for review.",
      };
      setMessages(prev => [...prev, confirmationMessage]);

    } catch (error) {
      console.error('Failed to submit feedback:', error);
      const errorMessage = {
        id: `err-fb-${Date.now()}`,
        type: 'bot',
        text: "Sorry, I couldn't log your feedback. Please try again later.",
        isError: true,
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setFeedbackContext(null);
      setFeedbackInput('');
    }
  }, [feedbackInput, feedbackContext]);

  // --- Handle Feedback Button Clicks ---
  const handleFeedbackClick = useCallback((messageId, accepted, originalQuery) => {
    // 1. Mark the message as feedback_pending: null to hide buttons
    setMessages(prev =>
      prev.map(msg =>
        msg.id === messageId ? { ...msg, feedback_pending: null } : msg
      )
    );

    if (accepted) {
      // 2. If accepted, just log it and show a thank you
      fetch(FEEDBACK_API_ENDPOINT, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: originalQuery, feedback_reason: "Accepted" }),
      }).catch(err => console.error("Failed to log acceptance:", err)); // Fire-and-forget

      const confirmationMessage = {
        id: `bot-fb-${Date.now()}`,
        type: 'bot',
        text: "Great! Recommendation accepted.",
      };
      setMessages(prev => [...prev, confirmationMessage]);

    } else {
      // 3. If rejected, open the feedback input form
      setFeedbackContext({ messageId, originalQuery });
    }
  }, []);


  return (
    <div className="flex h-screen bg-gray-900 text-gray-100 font-sans">
      
      {/* --- 1. Sidebar --- */}
      <aside className="w-80 bg-gray-950 p-6 flex flex-col flex-shrink-0">
        <div className="flex items-center gap-3 mb-8">
          <Brain className="w-8 h-8 text-blue-400" />
          <h1 className="text-2xl font-bold text-white">Strategist AI</h1>
        </div>
        <h2 className="text-sm font-semibold text-gray-400 uppercase mb-4">Our Goal</h2>
        <p className="text-sm text-gray-300 mb-6">
          Optimize vehicle pricing to minimize loss (Gradient Boosting) and maximize profit by stocking high-demand models (Random Forest).
        </p>
        <h3 className="text-sm font-semibold text-gray-400 uppercase mb-4">Agent Capabilities</h3>
        <ul className="space-y-3 text-sm">
          <li className="flex items-center gap-3 text-gray-300"><DollarSign className="w-4 h-4 text-green-400" /> Price/Discount Report</li>
          <li className="flex items-center gap-3 text-gray-300"><TrendingUp className="w-4 h-4 text-blue-400" /> Demand Forecast</li>
          <li className="flex items-center gap-3 text-gray-300"><BarChart2 className="w-4 h-4 text-yellow-400" /> Price History Graph</li>
          <li className="flex items-center gap-3 text-gray-300"><Newspaper className="w-4 h-4 text-orange-400" /> Recent News</li>
          <li className="flex items-center gap-3 text-gray-300"><ThumbsUp className="w-4 h-4 text-red-400" /> Social Sentiment</li>
          <li className="flex items-center gap-3 text-gray-300"><Shield className="w-4 h-4 text-indigo-400" /> Competitor Pricing</li>
          <li className="flex items-center gap-3 text-gray-300"><Globe className="w-4 h-4 text-teal-400" /> Live Web Search</li>
        </ul>
      </aside>

      {/* --- 2. Main Chat Area --- */}
      <main className="flex-1 flex flex-col h-screen">
        <header className="bg-gray-800 border-b border-gray-700 p-4 text-center">
          <h2 className="text-xl font-semibold text-white">Inventory Chat Console</h2>
        </header>

        {/* --- Message List --- */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {messages.map((msg) => (
            <ChatMessage 
              key={msg.id} 
              message={msg} 
              onFeedbackClick={handleFeedbackClick} 
            />
          ))}
          {isLoading && <LoadingMessage />}
          <div ref={endOfMessagesRef} />
        </div>

        {/* --- Footer / Input Area --- */}
        <footer className="p-6 bg-gray-800 border-t border-gray-700">
          {/* --- Feedback Input Form (Conditional) --- */}
          {feedbackContext && (
            <form onSubmit={handleFeedbackSubmit} className="mb-4 p-4 bg-gray-700 rounded-lg flex items-center gap-3">
              <input
                type="text"
                value={feedbackInput}
                onChange={(e) => setFeedbackInput(e.target.value)}
                placeholder="What's the correct price or info?"
                className="flex-1 bg-gray-600 border border-gray-500 rounded-md px-4 py-2 text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
              <button
                type="submit"
                className="bg-blue-600 hover:bg-blue-700 text-white font-semibold px-4 py-2 rounded-md flex items-center gap-2 transition-colors"
              >
                <Send className="w-4 h-4" /> Submit Feedback
              </button>
              <button
                type="button"
                onClick={() => setFeedbackContext(null)}
                className="bg-gray-500 hover:bg-gray-600 text-white font-semibold px-4 py-2 rounded-md transition-colors"
              >
                <X className="w-4 h-4" />
              </button>
            </form>
          )}

          {/* --- Main Chat Input Form --- */}
          <form onSubmit={handleSendMessage} className="flex items-center gap-3">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask for a discount, demand report, or web search..."
              className="flex-1 bg-gray-700 border border-gray-600 rounded-md px-4 py-3 text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
              disabled={isLoading || !!feedbackContext}
            />
            <button
              type="submit"
              className="bg-blue-600 hover:bg-blue-700 text-white font-semibold px-5 py-3 rounded-md flex items-center gap-2 transition-colors disabled:opacity-50"
              disabled={isLoading || !!feedbackContext}
            >
              <ArrowUp className="w-5 h-5" />
            </button>
          </form>
        </footer>
      </main>
    </div>
  );
}

// --- 2. Chat Message Component ---
function ChatMessage({ message, onFeedbackClick }) {
  const isBot = message.type === 'bot';

  if (message.isError) {
    return (
      <div className="flex items-start gap-3">
        <div className="flex-shrink-0 bg-red-800 rounded-full p-2">
          <AlertTriangle className="w-5 h-5 text-red-300" />
        </div>
        <div className="bg-red-900 border border-red-700 rounded-lg px-4 py-3 max-w-2xl">
          <p className="text-red-200">{message.text}</p>
        </div>
      </div>
    );
  }

  return (
    <div className={`flex items-start gap-3 ${isBot ? '' : 'flex-row-reverse'}`}>
      {isBot && (
        <div className="flex-shrink-0 bg-gray-700 rounded-full p-2">
          <MessageSquare className="w-5 h-5 text-blue-300" />
        </div>
      )}
      
      {/* --- Message Bubble --- */}
      <div 
        className={`rounded-lg px-5 py-3 max-w-2xl ${
          isBot 
            ? 'bg-gray-800 border border-gray-700' 
            : 'bg-blue-600 text-white'
        }`}
      >
        {/* --- Render Markdown Text --- */}
        <MarkdownRenderer content={message.text} />
        
        {/* --- Render Visualizations --- */}
        <SpecificReportRenderer 
          report_data={message.report_data}
          chart_data={message.chart_data}
        />
        
        {/* --- Render Feedback Buttons (if pending) --- */}
        {message.feedback_pending && (
          <div className="mt-4 pt-3 border-t border-gray-600 flex items-center gap-3">
            <p className="text-sm text-gray-400 italic">Was this recommendation helpful?</p>
            <button
              onClick={() => onFeedbackClick(message.id, true, message.feedback_pending)}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-green-700 hover:bg-green-600 text-white text-sm font-medium transition-colors"
            >
              <Check className="w-4 h-4" /> Accept
            </button>
            <button
              onClick={() => onFeedbackClick(message.id, false, message.feedback_pending)}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-red-700 hover:bg-red-600 text-white text-sm font-medium transition-colors"
            >
              <X className="w-4 h-4" /> Reject
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

// --- 3. Loading Skeleton ---
function LoadingMessage() {
  return (
    <div className="flex items-start gap-3">
      <div className="flex-shrink-0 bg-gray-700 rounded-full p-2">
        <MessageSquare className="w-5 h-5 text-blue-300" />
      </div>
      <div className="bg-gray-800 border border-gray-700 rounded-lg px-5 py-3 max-w-xs">
        <div className="animate-pulse flex space-x-2">
          <div className="rounded-full bg-gray-600 h-2.5 w-2.5"></div>
          <div className="rounded-full bg-gray-600 h-2.5 w-2.5"></div>
          <div className="rounded-full bg-gray-600 h-2.5 w-2.5"></div>
        </div>
      </div>
    </div>
  );
}

// --- 4. Report & Graph Renderer ---
// This component decides which report (if any) to render
function SpecificReportRenderer({ report_data, chart_data }) {
  // --- A. Render Graph ---
  if (chart_data?.image_data) {
    return <GraphVisualization data={chart_data} />;
  }

  // --- B. Render Structured Report ---
  if (!report_data?.intent) return null;

  switch (report_data.intent) {
    case 'get_discount':
      return <DiscountReport data={report_data} />;
    case 'get_demand':
      return <DemandReport data={report_data} />;
    case 'get_news':
      return <NewsReport data={report_data} />;
    case 'get_sentiment':
      return <SentimentReport data={report_data} />;
    case 'get_competitor_pricing':
      return <CompetitorReport data={report_data} />;
    case 'get_web_search': // <-- NEW
      return <WebSearchReport data={report_data} />;
    default:
      return null;
  }
}

// --- 5. Visualization Components ---

function GraphVisualization({ data }) {
  const { title, image_data } = data;
  const imageUrl = `data:image/png;base64,${image_data}`;

  return (
    <div className="mt-4 p-4 bg-gray-900 border border-gray-700 rounded-lg">
      <h4 className="text-lg font-semibold text-white mb-3">{title}</h4>
      <img 
        src={imageUrl} 
        alt={title} 
        className="w-full h-auto rounded-md" 
      />
      <a
        href={imageUrl}
        download={`${title.replace(/ /g, '_')}.png`}
        className="inline-block mt-4 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white text-sm font-medium rounded-md transition-colors"
      >
        Download Plot
      </a>
    </div>
  );
}

function DiscountReport({ data }) {
  return (
    <div className="mt-4 p-4 bg-gray-900 border border-gray-700 rounded-lg">
      <h4 className="text-lg font-semibold text-green-400 mb-3">Full Discount Report: {data.car_identified}</h4>
      <div className="grid grid-cols-2 gap-4 text-sm">
        <InfoBlock title="Days in Inventory" value={data.days_in_inventory} unit="days" />
        <InfoBlock title="Base Price" value={data.base_price} />
        <InfoBlock title="Recommended Discount" value={data.recommended_discount_pct} unit="%" />
        <InfoBlock title="Predicted Sale Price" value={data.predicted_sale_price} />
        <InfoBlock title="Sentiment (r/cars)" value={data.external_sentiment?.sentiment_score} unit="/ 1.0" />
        <InfoBlock title="Local Avg. Price" value={data.competitor_analysis?.local_avg_price} />
      </div>
    </div>
  );
}

function DemandReport({ data }) {
  return (
    <div className="mt-4 p-4 bg-gray-900 border border-gray-700 rounded-lg">
      <h4 className="text-lg font-semibold text-blue-400 mb-3">30-Day Demand Forecast</h4>
      <div className="space-y-2">
        <DemandRow title="High Demand" models={data.high_demand} color="text-green-400" />
        <DemandRow title="Medium Demand" models={Tta.medium_demand} color="text-yellow-400" />
        <DemandRow title="Low Demand" models={data.low_demand} color="text-red-400" />
      </div>
    </div>
  );
}

function NewsReport({ data }) {
  return (
    <div className="mt-4 p-4 bg-gray-900 border border-gray-700 rounded-lg">
      <h4 className="text-lg font-semibold text-orange-400 mb-3">Recent News</h4>
      <ul className="space-y-2">
        {data.articles?.map((article, index) => (
          <li key={index} className="text-sm">
            <a href={article.url} target="_blank" rel="noopener noreferrer" className="text-blue-300 hover:underline">{article.title}</a>
            <span className="text-gray-400 text-xs ml-2">({article.source})</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

function SentimentReport({ data }) {
  return (
    <div className="mt-4 p-4 bg-gray-900 border border-gray-700 rounded-lg">
      <h4 className="text-lg font-semibold text-red-400 mb-3">Social Sentiment ({data.source})</h4>
      <div className="flex items-center gap-4">
        <div className="text-4xl font-bold text-red-300">{data.sentiment_score} <span className="text-lg">/ 1.0</span></div>
        <p className="text-sm text-gray-300 flex-1">{data.summary}</p>
      </div>
    </div>
  );
}

function CompetitorReport({ data }) {
  return (
    <div className="mt-4 p-4 bg-gray-900 border border-gray-700 rounded-lg">
      <h4 className="text-lg font-semibold text-indigo-400 mb-3">Competitor Pricing Analysis</h4>
      <div className="mb-3">
        <InfoBlock title="Local Avg. Price" value={data.local_avg_price} />
      </div>
      <table className="w-full text-sm text-left">
        <thead className="text-xs text-gray-400 uppercase bg-gray-800">
          <tr>
            <th scope="col" className="px-4 py-2">Competitor</th>
            <th scope="col" className="px-4 py-2">Reported Price</th>
          </tr>
        </thead>
        <tbody>
          {data.competitors?.map((comp, index) => (
            <tr key={index} className="border-b border-gray-700">
              <td className="px-4 py-2 font-medium text-gray-200">{comp.dealer}</td>
              <td className="px-4 py-2 text-gray-300">{comp.price}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function WebSearchReport({ data }) {
  return (
    <div className="mt-4 p-4 bg-gray-900 border border-gray-700 rounded-lg">
      <h4 className="text-lg font-semibold text-teal-400 mb-3">Live Web Search Results</h4>
      <ul className="space-y-3">
        {data.search_results?.map((result, index) => (
          <li key={index} className="text-sm">
            <a href={result.link} target="_blank" rel="noopener noreferrer" className="text-blue-300 hover:underline text-base font-medium">{result.title}</a>
            <p className="text-gray-400 text-xs mt-0.5">{result.snippet}</p>
            <span className="text-gray-500 text-xs">{result.source}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}


// --- 6. Helper Components ---

function InfoBlock({ title, value, unit = '' }) {
  return (
    <div className="bg-gray-800 p-3 rounded-md">
      <h5 className="text-xs text-gray-400 uppercase font-semibold">{title}</h5>
      <p className="text-xl font-bold text-white">
        {value}
        {unit && <span className="text-base font-normal ml-1">{unit}</span>}
      </p>
    </div>
  );
}

function DemandRow({ title, models = [], color = 'text-white' }) {
  return (
    <div className="flex items-center text-sm">
      <span className={`font-semibold w-28 ${color}`}>{title}:</span>
      <span className="text-gray-300">{models.join(', ')}</span>
    </div>
  );
}

// Simple Markdown Renderer (for bold and newlines)
function MarkdownRenderer({ content }) {
  if (!content) return null;

  const renderPart = (part, index) => {
    // Match **bold**
    if (part.startsWith('**') && part.endsWith('**')) {
      return <strong key={index}>{part.substring(2, part.length - 2)}</strong>;
    }
    // Match *italic*
    if (part.startsWith('*') && part.endsWith('*')) {
      return <em key={index}>{part.substring(1, part.length - 1)}</em>;
    }
    return <span key={index}>{part}</span>;
  };

  const lines = content.split('\n');
  
  return (
    <div>
      {lines.map((line, lineIndex) => (
        <p key={lineIndex} className="my-1">
          {line.split(/(\*\*.*?\*\*|\*.*?\*)/g).filter(Boolean).map(renderPart)}
        </p>
      ))}
    </div>
  );
}