# Vortex - AI-Powered Strategy Agent for Car Dealerships

**Vortex** is an intelligent conversational agent designed to help car dealership sales executives with real-time pricing, demand forecasting, and market insights. It combines machine learning models, live APIs, and LLM reasoning to deliver actionable recommendations.

## Features

### 🎯 Core Capabilities

- **Price & Discount Recommendations**: Intelligent pricing using gradient boosting models based on inventory age and competitor data
- **Demand Forecasting**: 30-day demand predictions using Random Forest and SARIMAX models
- **Price History Graphs**: Visual price trends over time for inventory analysis
- **Market Intelligence**: Real-time news aggregation and competitor pricing analysis
- **Social Sentiment Analysis**: Reddit sentiment analysis for vehicle models using VADER
- **Live Web Search**: General-purpose web search for market insights and inquiries
- **User Feedback System**: Closed-loop feedback for continuous model improvement

### 🔄 Agent Capabilities

The AI Agent can handle the following intents:
- `get_discount` - Price & discount recommendations for specific cars
- `get_demand` - General market sales trends and demand forecasts
- `get_graph` - Price history visualizations
- `get_news` - Recent automotive news
- `get_sentiment` - Social media sentiment (Reddit)
- `get_competitor_pricing` - Competitor price analysis
- `get_web_search` - General web search queries

## Project Architecture

```
Vortex-main/
├── Frontend/                    # React + Vite UI
│   ├── src/
│   │   ├── App.jsx             # Main chat interface with feedback system
│   │   ├── App.css
│   │   └── main.jsx
│   ├── package.json
│   ├── vite.config.js
│   └── tailwind.config.js
│
├── Backend/                     # FastAPI server
│   ├── main.py                 # Core API endpoints & routing
│   ├── ml_models.py            # Hybrid demand forecasting model
│   ├── final_processed_cars.csv # Vehicle inventory database
│   └── firebase-service-account.json (optional)
│
├── fast.py                      # Utility script
└── README.md
```

## Tech Stack

### Frontend
- **React 18** - UI framework
- **Vite** - Build tool
- **Tailwind CSS** - Styling
- **Lucide React** - Icons
- **Axios/Fetch API** - HTTP client

### Backend
- **FastAPI** - Web framework
- **Pandas** - Data processing
- **Scikit-learn** - Machine learning (Random Forest, preprocessing)
- **Statsmodels** - Time series forecasting (SARIMAX)
- **VADER Sentiment** - Sentiment analysis
- **Firebase Firestore** - Feedback logging
- **Matplotlib** - Graph generation

### External APIs
- **OpenRouter** - LLM inference (Google Gemini 2.0 Flash)
- **NewsAPI** - Automotive news aggregation
- **Google Custom Search** - Web search capability
- **Reddit API** - Social sentiment analysis

## Setup Instructions

### Prerequisites
- Python 3.8+ 
- Node.js 16+
- Environment variables file (`.env`)

### Environment Variables

Create a `.env` file in the `Backend` folder:

```env
OPENROUTER_API_KEY=your_openrouter_key
NEWS_API_KEY=your_newsapi_key
GOOGLE_SEARCH_API_KEY=your_google_search_key
GOOGLE_SEARCH_CX_ID=your_custom_search_engine_id
```

**Optional**: For feedback logging, add your Firebase service account JSON file as `Backend/firebase-service-account.json`

### Backend Setup

```bash
cd Backend

# Create virtual environment (recommended)
python -m venv venv
source venv/Scripts/activate  # Windows
# or
source venv/bin/activate      # macOS/Linux

# Install dependencies
pip install fastapi uvicorn pandas scikit-learn statsmodels vadersentiment firebase-admin python-dotenv matplotlib httpx

# Run the server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Backend runs at: `http://localhost:8000`

### Frontend Setup

```bash
cd Frontend

# Install dependencies
npm install

# Run development server
npm run dev
```

Frontend runs at: `http://localhost:5173` (default Vite port)

## API Endpoints

### POST `/api/chat`
Main chat endpoint for handling user queries.

**Request:**
```json
{
  "query": "What discount should I offer for a Toyota Innova with 50k km?"
}
```

**Response:**
```json
{
  "response_text": "...",
  "report_data": { /* structured report */ },
  "chart_data": { /* optional graph data */ }
}
```

### POST `/api/feedback`
Log user feedback for model improvement.

**Request:**
```json
{
  "query": "Original user query",
  "feedback_reason": "The recommended price was too high"
}
```

## Usage Example

1. **Start Backend**: `uvicorn main:app --reload`
2. **Start Frontend**: `npm run dev`
3. **Open Chat**: Navigate to the UI and start asking questions

Example queries:
- "What discount should I offer for a Toyota Fortuner?"
- "What cars are selling well in the next 30 days?"
- "Show me the price history for a Honda City"
- "What's the competitor pricing for a Mahindra XUV700?"

## Model Architecture

### Hybrid Demand Forecasting
The system uses a **hybrid approach** combining:
- **Random Forest Regressor**: Feature-based demand prediction using car attributes (fuel type, transmission, age, mileage)
- **SARIMAX**: Time-series forecasting per brand for seasonal trends

### Performance Metrics
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² Score

## File Structure

| File | Purpose |
|------|---------|
| `main.py` | API routes, intent classification, agent logic |
| `ml_models.py` | Hybrid demand model training & evaluation |
| `App.jsx` | Chat UI, message rendering, feedback UI |
| `final_processed_cars.csv` | Vehicle inventory with features |

## Troubleshooting

### "API Key Missing" Error
Ensure all required API keys are in the `.env` file and the Backend is restarted.

### "CSV not found" Error
Place `final_processed_cars.csv` in the `Backend` folder. The app will use mock data if missing.

### "Firebase not initialized"
Feedback logging will be disabled without `firebase-service-account.json`. The chat still works normally.

### CORS Issues
The Backend has CORS enabled for all origins. If issues persist, check FastAPI CORS settings.

## Future Improvements

- [ ] Inventory sync with real dealership databases
- [ ] Advanced NLP for context understanding
- [ ] Multi-user sessions with history persistence
- [ ] Mobile app for field executives
- [ ] Predictive maintenance insights
- [ ] Lead scoring integration

## Contributors

Built as a hackathon project for intelligent dealership operations.

## License

MIT License - Feel free to use and modify for your dealership!