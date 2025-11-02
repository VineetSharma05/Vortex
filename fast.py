import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import time
import random
import re
import httpx  # For making async API calls
import os     # For getting API key from environment
import json

# --- OpenRouter & API Key Configuration ---
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
NEWS_API_KEY = os.environ.get("NEWS_API_KEY") # <-- NEW API KEY
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
LLM_MODEL = "deepseek/deepseek-r1-0528-qwen3-8b:free"

# --- System Prompt for the AI Agent ---
# This is the most important part. It tells the LLM how to behave
# and to ONLY respond with JSON.
AGENT_SYSTEM_PROMPT = """
You are an intelligent routing agent for a car dealership.
Your job is to receive a query from a sales executive and classify it into one of the following intents.
You MUST respond with ONLY a valid JSON object and nothing else.

The possible intents are:
1. `get_discount`: When the user is asking about the price, discount, or details for a SPECIFIC car.
2. `get_demand`: When the user is asking about general sales trends, what's selling well, or demand forecasts.
3. `get_graph`: When the user is asking for a price history or graph for a SPECIFIC car.
4. `get_news`: (NEW) When the user asks for recent news about a car.
5. `get_sentiment`: (NEW) When the user asks for social media sentiment (e.g., from Reddit) about a car.
6. `get_competitor_pricing`: (NEW) When the user asks for competitor prices for a car.
7. `general_report`: If the user's query is unclear, vague, or just a greeting.

For intents `get_discount`, `get_graph`, `get_news`, `get_sentiment`, and `get_competitor_pricing`,
you MUST extract the car's details.

Here are examples of query and expected JSON output:

Query: "Hi, what's the best discount you can give me on the 2023 Toyota Camry XSE?"
{"intent": "get_discount", "car_details": "2023 Toyota Camry XSE"}

Query: "What's been selling well this month?"
{"intent": "get_demand"}

Query: "Can you show me the price history for the blue Ford F-150?"
{"intent": "get_graph", "car_details": "blue Ford F-150"}

Query: "What's the news on the 2024 Mustang?"
{"intent": "get_news", "car_details": "2024 Mustang"}

Query: "What's the sentiment on r/cars for the Ioniq 5?"
{"intent": "get_sentiment", "car_details": "Ioniq 5"}

Query: "What are competitors charging for a 2022 RAV4?"
{"intent": "get_competitor_pricing", "car_details": "2022 RAV4"}

Query: "What's up?"
{"intent": "general_report"}
"""

# --- Pydantic Models (Data Contracts) ---
class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    response_text: str
    chart_data: dict | None = None
    report_data: dict | None = None

# --- Mock ML Model & External API Functions ---

# --- INTERNAL ML MODELS (from your team) ---

def mock_get_discount_prediction(car_details: str) -> dict:
    """ MOCK: Simulates calling the Gradient Boosting price prediction model. """
    time.sleep(1.0) # Simulate model inference
    days_in_inventory = random.randint(30, 150)
    base_price = 30000
    
    optimal_discount_pct = (days_in_inventory / 100) * random.uniform(1.0, 1.5)
    
    return {
        "car_identified": car_details.strip(),
        "days_in_inventory": days_in_inventory,
        "base_price": f"${base_price:,.2f}",
        "recommended_discount_pct": round(optimal_discount_pct, 2),
        "predicted_sale_price": f"${(base_price * (1 - (optimal_discount_pct / 100))):,.2f}"
    }

def mock_get_demand_forecast() -> dict:
    """ MOCK: Simulates calling the Random Forest demand forecast model. """
    time.sleep(0.5) 
    models = ["Honda Civic", "Toyota Camry", "Ford F-150", "Tesla Model Y", "Honda CR-V", "Toyota RAV4"]
    random.shuffle(models)
    return {
        "high_demand": models[0:2], "medium_demand": models[2:4], "low_demand": models[4:6],
        "forecast_period": "Next 30 Days", "key_drivers": ["Lower interest rates", "Seasonal demand (Spring)"]
    }

def mock_get_price_graph_data(car_details: str) -> dict:
    """ MOCK: Simulates querying a DB for historical price data. """
    time.sleep(0.2)
    labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
    prices = [32000, 31800, 31750, 31500, 31300, 31200]
    return { "title": f"Price History for {car_details.strip()}", "labels": labels, "data": prices }

# --- NEW: EXTERNAL API MOCKS (Sentiment, News, Competitors) ---

async def mock_get_news(car_details: str) -> dict:
    if NEWS_API_KEY:
        try:
            async with httpx.AsyncClient() as client:
                params = {"q": car_details, "apiKey": NEWS_API_KEY, "pageSize": 3, "sortBy": "relevancy"}
                response = await client.get("https://newsapi.org/v2/everything", params=params)
            articles = response.json().get("articles", [])
            return {
                "summary": f"Found {len(articles)} relevant articles.",
                "articles": [{"title": a['title'], "source": a['source']['name'], "url": a['url']} for a in articles]
            }
        except Exception as e:
            print(f"NewsAPI call failed: {e}")

    return {
        "summary": "Recent news is positive, highlighting new safety features.",
        "articles": [
            {"title": f"{car_details} wins 'Safety Pick of the Year'", "source": "CarNews.com"},
            {"title": f"Review: The {car_details} is a solid family choice", "source": "AutoTrader"}
        ]
    }

async def mock_get_reddit_sentiment(car_details: str) -> dict:
    """
    MOCK: Simulates calling your team's internal Reddit sentiment model.
    """
    time.sleep(0.3)
    score = random.uniform(0.5, 0.9)
    return {
        "source": "r/cars (Simulated)",
        "sentiment_score": round(score, 2),
        "summary": "Positive sentiment. Owners are praising its reliability and fuel economy. Some mentions of a 'boring' interior."
    }

async def mock_get_competitor_rates(car_details: str) -> dict:
    """
    MOCK: Simulates calling a competitor pricing API.
    """
    time.sleep(0.7)
    return {
        "local_avg_price": f"${30500 + random.randint(-500, 500):,.2f}",
        "competitors": [
            {"dealer": "City Auto", "price": "$30,450"},
            {"dealer": "Suburban Motors", "price": "$30,600"}
        ]
    }

# --- NEW: AI Agent Function (with new fallback logic) ---
async def get_llm_intent(query: str) -> dict:
    """ Calls the OpenRouter LLM to classify the user's intent. """
    if not OPENROUTER_API_KEY:
        print("OPENROUTER_API_KEY not set. Using simulation.")
        return simulate_intent(query)
        
    headers = { "Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost", "X-Title": "Hackathon Dealership AI" }
    
    body = { "model": LLM_MODEL, "messages": [{"role": "system", "content": AGENT_SYSTEM_PROMPT}, {"role": "user", "content": query}],
             "response_format": {"type": "json_object"} }
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(OPENROUTER_API_URL, headers=headers, json=body, timeout=20.0)
            
            if response.status_code == 200:
                json_content_str = response.json()['choices'][0]['message']['content']
                return json.loads(json_content_str)
            else:
                print(f"Error from OpenRouter: {response.status_code} - {response.text}")
                return {"intent": "general_report", "error": f"API error: {response.status_code}"}
                
    except Exception as e:
        print(f"Exception during LLM call: {e}")
        return {"intent": "general_report", "error": str(e)}

def simulate_intent(query: str) -> dict:
    """ Fallback simulation, now updated with new intents. """
    query = query.lower()
    car_match = re.search(r"(for|on|about) the ([\w\s\d-]+)", query)
    car_details = "Unknown Car"
    if car_match: car_details = car_match.group(2).strip()

    if "news" in query: return {"intent": "get_news", "car_details": car_details}
    if "sentiment" in query or "reddit" in query: return {"intent": "get_sentiment", "car_details": car_details}
    if "competitor" in query or "competing" in query: return {"intent": "get_competitor_pricing", "car_details": car_details}
    if "discount" in query or (car_match and "tell me about" in query): return {"intent": "get_discount", "car_details": car_details}
    if "demand" in query or "selling well" in query: return {"intent": "get_demand"}
    if "graph" in query or "history" in query: return {"intent": "get_graph", "car_details": car_details}
    
    return {"intent": "general_report"}

# --- FastAPI App Initialization ---

app = FastAPI(
    title="Dealership AI Agent API",
    description="API for the hackathon project, routing queries to ML models."
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# --- API Endpoints ---

@app.get("/")
def read_root(): return {"status": "DealSership AI Agent API is running."}

@app.post("/api/chat", response_model=ChatResponse)
async def chat_agent(request: ChatRequest):
    """
    This is your main "AI Agent" endpoint.
    It now handles the new intents for external data.
    """
    query = request.query
    
    # --- 1. Get Intent from LLM ---
    llm_response = await get_llm_intent(query)
    intent = llm_response.get("intent", "general_report")
    car_details = llm_response.get("car_details", "Unknown Car")
    
    if "error" in llm_response:
        print("LLM call failed, falling back to simulation.")
        llm_response = simulate_intent(query)
        intent = llm_response.get("intent", "general_report")
        car_details = llm_response.get("car_details", "Unknown Car")
        
    print(f"Query: '{query}' -> Intent: '{intent}', Car: '{car_details}'")

    # --- 2. Tool/Model Calling (Based on LLM-derived intent) ---
    
    if intent == "get_discount":
        # This is the "Full Report" intent. It calls ALL services.
        
        # Call internal ML models
        report_data = mock_get_discount_prediction(car_details) # Price model
        demand_data = mock_get_demand_forecast()                 # Demand model
        
        # Call external APIs
        sentiment_data = await mock_get_reddit_sentiment(car_details)
        news_data = await mock_get_news(car_details)
        competitor_data = await mock_get_competitor_rates(car_details)
        
        # Combine all data into one report
        report_data['demand_forecast'] = demand_data
        report_data['external_sentiment'] = sentiment_data
        report_data['recent_news_summary'] = news_data.get('summary', 'No news found.')
        report_data['competitor_analysis'] = competitor_data
        
        # --- TODO: Call LLM *again* to synthesize a final summary ---
        # For now, we'll just create a simple string.
        response_text = f"Here's the full **Discount Report** for the **{report_data['car_identified']}**:\n\n"
        response_text += f"**Price Recommendation:**\n"
        response_text += f"- It's been in stock for **{report_data['days_in_inventory']} days**.\n"
        response_text += f"- I recommend a **{report_data['recommended_discount_pct']}%** discount, bringing the price to **{report_data['predicted_sale_price']}**.\n\n"
        response_text += f"**Justification:**\n"
        response_text += f"- **Demand:** This model is in **{demand_data['high_demand'][0]}**-level demand (similar to {demand_data['high_demand'][1]}).\n"
        response_text += f"- **Sentiment:** Public sentiment is **{sentiment_data['sentiment_score']}/1.0** (Source: {sentiment_data['source']}).\n"
        response_text += f"- **News:** {news_data['summary']}\n"
        response_text += f"- **Competitors:** Local average is **{competitor_data['local_avg_price']}**."

        return ChatResponse(response_text=response_text, report_data=report_data)

    elif intent == "get_demand":
        report_data = mock_get_demand_forecast()
        response_text = f"Here's the 30-day demand forecast: \n- **High Demand:** {', '.join(report_data['high_demand'])} \n- **Low Demand:** {', '.join(report_data['low_demand'])}. \nWe should focus on stocking the high-demand models."
        return ChatResponse(response_text=response_text, report_data=report_data)

    elif intent == "get_graph":
        chart_data = mock_get_price_graph_data(car_details)
        response_text = f"Here is the price history graph for the {car_details}."
        return ChatResponse(response_text=response_text, chart_data=chart_data)

    # --- NEW: Handle specific external API queries ---
    elif intent == "get_news":
        report_data = await mock_get_news(car_details)
        response_text = f"Here's the recent news for the {car_details}:\n- "
        response_text += "\n- ".join([a['title'] + f" (*{a['source']}*)" for a in report_data['articles']])
        return ChatResponse(response_text=response_text, report_data=report_data)

    elif intent == "get_sentiment":
        report_data = await mock_get_reddit_sentiment(car_details)
        response_text = f"Here's the sentiment for the {car_details} from {report_data['source']}:\n"
        response_text += f"- **Score:** {report_data['sentiment_score']}/1.0\n"
        response_text += f"- **Summary:** {report_data['summary']}"
        return ChatResponse(response_text=response_text, report_data=report_data)

    elif intent == "get_competitor_pricing":
        report_data = await mock_get_competitor_rates(car_details)
        response_text = f"Here's the competitor pricing for the {car_details}:\n"
        response_text += f"- **Local Average:** {report_data['local_avg_price']}\n- **Competitors:**\n"
        response_text += "\n".join([f"  - {c['dealer']}: {c['price']}" for c in report_data['competitors']])
        return ChatResponse(response_text=response_text, report_data=report_data)

    else: # Catches "general_report"
        response_text = "I can help with discount recommendations, demand forecasts, or competitor/news lookups. What would you like to know?"
        return ChatResponse(response_text=response_text)