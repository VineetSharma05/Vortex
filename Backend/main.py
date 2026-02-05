import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import time
import random
import re
import httpx
import os
import json
import io
import base64
from datetime import datetime
from pathlib import Path
import sys
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd # <-- For ML model

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import ml_models    # <-- Your REAL demand model

# --- Matplotlib Setup (for non-interactive backend) ---
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- Firebase Admin SDK Setup ---
import firebase_admin
from firebase_admin import credentials, firestore

# --- Load Environment Variables ---
from dotenv import load_dotenv
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
GOOGLE_SEARCH_API_KEY = os.getenv("GOOGLE_SEARCH_API_KEY") # <-- NEW
GOOGLE_SEARCH_CX_ID = os.getenv("GOOGLE_SEARCH_CX_ID") # <-- NEW

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
LLM_MODEL = "google/gemini-2.0-flash-001" # Use a preferred model

# --- NEW: Demand Multipliers ---
DEMAND_MULTIPLIERS = {
    'High': 1.05,    # 5% premium for high demand
    'Good': 1.02,    # 2% premium for good demand
    'Mid': 1.00,     # No adjustment for mid demand
    'Low': 0.97,     # 3% discount for low demand
    'No': 0.92      # 8% discount for no demand
}

# --- NEW: Load the REAL CSV database on startup ---
CSV_FILE_PATH = BASE_DIR / 'final_processed_cars.csv' # Make sure this file is in the same directory
CAR_DATABASE_DF = None
try:
    CAR_DATABASE_DF = pd.read_csv(CSV_FILE_PATH)
    # Ensure key columns are lowercase for easier matching
    CAR_DATABASE_DF.columns = [col.lower() for col in CAR_DATABASE_DF.columns]
    print(f"✅ Successfully loaded '{CSV_FILE_PATH}' into memory.")
    print(f"   -> Found {len(CAR_DATABASE_DF)} car records.")
except FileNotFoundError:
    print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(f"WARNING: '{CSV_FILE_PATH}' not found.")
    print(f"         The app will rely on 100% mock data.")
    print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
except Exception as e:
    print(f"Error loading CSV: {e}")

# --- Firebase Initialization ---
try:
    # Path to your service account key JSON file
    cred_path = BASE_DIR / 'firebase-service-account.json'
    if not cred_path.exists():
        print("WARNING: 'firebase-service-account.json' not found. Feedback loop will not work.")
        db = None
    else:
        cred = credentials.Certificate(str(cred_path))
        firebase_admin.initialize_app(cred)
        db = firestore.client()
        print("Firebase Firestore initialized successfully.")
except Exception as e:
    print(f"Error initializing Firebase: {e}")
    db = None

# --- System Prompt for the AI Agent (Kept from your new file) ---
AGENT_SYSTEM_PROMPT = """
You are an intelligent routing agent for a car dealership.
Your job is to receive a query from a sales executive and classify it into one of the following intents.
You MUST respond with ONLY a valid JSON object and nothing else.

The possible intents are:
1. `get_discount`: When the user is asking about the price, discount, or details for a SPECIFIC car.
2. `get_demand`: When the user is asking about general sales trends, what's selling well, or demand forecasts.
3. `get_graph`: When the user is asking for a price history or graph for a SPECIFIC car.
4. `get_news`: When the user asks for recent news about a car.
5. `get_sentiment`: When the user asks for social media sentiment (e.g., from Reddit) about a car.
6. `get_competitor_pricing`: When the user asks for competitor prices for a car.
7. `get_web_search`: (NEW) When the user asks a general question, for a definition, or a query that doesn't fit other categories (e.g., "what is the new 2025 EPA standard?").
8. `general_report`: If the user's query is unclear, vague, or just a greeting.

For intents `get_discount`, `get_graph`, `get_news`, `get_sentiment`, `get_competitor_pricing`, and `get_web_search`,
you MUST extract the `car_details` or `search_query` string.

(Examples are unchanged)
...
"""

# --- Pydantic Models (Data Contracts) ---
class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    response_text: str
    chart_data: dict | None = None
    report_data: dict | None = None

class FeedbackRequest(BaseModel):
    query: str
    feedback_reason: str

# --- Sentiment Analyzer (Unchanged) ---
vader_analyzer = SentimentIntensityAnalyzer()


# --- REAL CSV DATABASE FUNCTION ---
def get_car_features_from_csv_data(car_details: str) -> dict:
    """
    REAL: Searches the loaded CSV DataFrame for the car's features.
    This replaces the unreliable Google Search or mock string.
    """
    print(f"Running: get_car_features_from_csv_data for '{car_details}'")
    
    car_data = {}
    data_source = "Mock Fallback"

    if CAR_DATABASE_DF is not None:
        try:
            # --- 1. Create a robust regex to find the car line ---
            # e.g., "mercedes benz gle" -> ".*mercedes.*benz.*gle.*"
            # This is a safer way to match "dvn" (description)
            search_terms = car_details.lower().split()
            regex_pattern = ".*" + ".*".join(re.escape(term) for term in search_terms) + ".*"
            
            # Search in 'dvn' column (as seen in your CSV data)
            mask = CAR_DATABASE_DF['dvn'].str.contains(regex_pattern, case=False, na=False)
            
            if mask.any():
                data_source = "Real CSV Data"
                # --- 2. Get the *first* matching row as a dictionary ---
                found_row = CAR_DATABASE_DF[mask].iloc[0].to_dict()
                print(f"  -> Found match in CSV: {found_row.get('dvn')}")

                # --- 3. Extract features needed by the demand model ---
                # (Based on Untitled4.ipynb feature list)
                year = int(found_row.get('year', 2018))
                km = float(found_row.get('kilometers_driven', 50000))
                selling_price = float(found_row.get('selling_price', 800000))

                car_data = {
                    "Name": found_row.get('name', 'maruti'), # Get brand, fallback to maruti
                    "Year": year,
                    "Selling_Price": selling_price,
                    "Kilometers_Driven": km,
                    "Fuel_Type": found_row.get('fuel_type', 'petrol').capitalize(),
                    "Transmission": found_row.get('transmission', 'manual').capitalize(),
                    "car_age": 2024 - year,
                    "source": data_source
                }
                print(f"  -> Parsed: Year={year}, KM={km}, Price=₹{selling_price:,.2f}")
            
        except Exception as e:
            print(f"  -> Error while searching CSV: {e}. Reverting to fallback.")
            car_data = {} # Ensure fallback is triggered

    # --- 4. Fallback if no CSV line was matched or CSV failed to load ---
    if not car_data:
        print("  -> No match in CSV, using smart mock fallback.")
        year = random.randint(2018, 2023)
        km = random.randint(20000, 80000)
        
        if any(keyword in car_details.lower() for keyword in ["benz", "mercedes", "audi", "bmw", "gle", "gla"]):
            base_price = 7000000
            selling_price = max(base_price - (2024 - year) * 500000 - (km * 2), 2000000)
        else:
            base_price = 1200000
            selling_price = max(base_price - (2024 - year) * 100000 - (km / 2), 300000)
            
        car_data = {
            "Name": "maruti", "Year": year, "Selling_Price": selling_price,
            "Kilometers_Driven": km, "Fuel_Type": "Petrol", "Transmission": "Manual",
            "car_age": 2024 - year, "source": data_source
        }

    # --- 5. Mock Feature Engineering & Encoding (from notebook) ---
    features = car_data.copy()
    
    # Use the same maps from your notebook
    brand_map = {"maruti": 29, "honda": 15, "hyundai": 17, "tata": 30, "mercedes-benz": 31} # Added merc
    fuel_map = {"Petrol": 0, "Diesel": 1, "Cng": 2} 
    trans_map = {"Manual": 0, "Automatic": 1}

    features["brand_encoded"] = brand_map.get(features["Name"].lower(), 0) # match lowercase
    features["Fuel_Type"] = fuel_map.get(features["Fuel_Type"], 0)
    features["Transmission"] = trans_map.get(features["Transmission"], 0)

    features["price_per_km"] = features["Selling_Price"] / (features["Kilometers_Driven"] + 1)
    features["price_per_km"] = features["price_per_km"] / 10 # Mock scaling
    
    features["price_x_age"] = features["price_per_km"] * features["car_age"]
    features["price_x_mileage"] = features["price_per_km"] * features["Kilometers_Driven"]
    features["age_x_mileage"] = features["car_age"] * features["Kilometers_Driven"]

    print(f"DB Features for {car_details}: Found {len(features)} features (Source: {car_data['source']}).")
    return features


# --- NEW: Helper function for demand ---
def _map_score_to_demand(score: float) -> str:
    """Maps a 0.0-1.0 demand score to a category."""
    if score > 0.8: return 'High'
    if score > 0.6: return 'Good'
    if score > 0.4: return 'Mid'
    if score > 0.2: return 'Low'
    return 'No'

# --- Mock ML Model & External API Functions (CURRENCY FIXED) ---

def mock_get_discount_prediction(
    car_details: str, 
    car_features: dict, 
    demand_score: float, 
    competitor_data: dict
) -> dict:
    """ 
    SMART MOCK: Simulates the price model using real inputs.
    This is the "nearly placeholdered" logic requested.
    """
    print("Running: mock_get_discount_prediction (Smarter Logic)")
    
    days_in_inventory = random.randint(30, 150)
    
    # --- THIS IS NOW MORE RELIABLE ---
    base_price = car_features.get("Selling_Price", 800000) # Get price from CSV or fallback
    
    # 1. Try to parse a competitor price from snippets
    competitor_price = base_price * 1.03 # Default to 3% higher
    competitor_price_found = "Not found, using default."
    snippet_text = json.dumps(competitor_data['competitors'])
    # --- FIX: Stricter regex to match numbers with commas (e.g., "1,20.5" or "5.5") not "..." ---
    lakh_match = re.search(r'₹?\s*([\d,]+(?:\.[\d]+)?)\s*(Lakh|L)', snippet_text, re.IGNORECASE)
    if lakh_match:
        try:
            # --- FIX: Remove commas before converting to float ---
            competitor_price = float(lakh_match.group(1).replace(',', '')) * 100000
            competitor_price_found = f"₹{competitor_price:,.2f} (from web snippet)"
            print(f"  -> Parsed competitor price: {competitor_price_found}")
        except ValueError:
             print("  -> Regex matched non-float, using default price.")
             # Keep default competitor_price
    else:
        print("  -> Could not parse competitor price, using default.")

    # 2. Get demand category and multiplier
    demand_category = _map_score_to_demand(demand_score)
    demand_multiplier = DEMAND_MULTIPLIERS[demand_category]
    print(f"  -> Demand Score: {demand_score} -> Category: {demand_category} (x{demand_multiplier})")

    # 3. Plausible "Placeholder" Formula
    # Anchor to the lowest price (ours or competitor's)
    price_anchor = min(base_price, competitor_price)
    
    # Adjust for demand (high demand can fetch a premium)
    demand_adjusted_price = price_anchor * demand_multiplier
    
    # Simulate inventory cost (e.g., 0.05% per day, max 10%)
    inventory_discount_factor = min((days_in_inventory * 0.0005), 0.10)
    
    # Final price is demand-adjusted, then discounted for inventory age
    predicted_price = demand_adjusted_price * (1 - inventory_discount_factor)
    
    # Calculate discount % based on *our* original base_price
    recommended_discount_pct = (base_price - predicted_price) / base_price * 100
    
    return {
        "intent": "get_discount",
        "car_identified": car_details.strip(),
        "days_in_inventory": days_in_inventory,
        "base_price": f"₹{base_price:,.2f}",
        "data_source": car_features.get("source", "Unknown"),
        "competitor_price_found": competitor_price_found,
        "demand_category": demand_category,
        "demand_multiplier": demand_multiplier,
        "inventory_discount_factor": round(inventory_discount_factor, 3),
        "recommended_discount_pct": round(recommended_discount_pct, 2),
        "predicted_sale_price": f"₹{predicted_price:,.2f}"
    }

def mock_get_demand_forecast() -> dict:
    """ 
    MOCK: Simulates a GENERAL market analysis.
    This is now DIFFERENT from your specific car demand model.
    We will keep this to answer the "what's selling well?" query.
    """
    time.sleep(0.4)
    models = ["Maruti Suzuki Swift", "Hyundai Creta", "Tata Nexon", "Kia Seltos", "Mahindra XUV700", "Tata Punch"]
    random.shuffle(models)
    return {
        "intent": "get_demand",
        "high_demand": models[0:2],
        "medium_demand": models[2:4],
        "low_demand": models[4:6],
        "forecast_period": "Next 30 Days",
        "key_drivers": ["Festive season approaching", "Stable fuel prices"]
    }

def mock_get_price_graph_data(car_details: str) -> dict:
    """ MOCK: Simulates querying a DB and generating a Matplotlib graph (INR). """
    time.sleep(0.3)
    labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
    
    # --- NEW: Smarter graph prices ---
    if any(keyword in car_details.lower() for keyword in ["benz", "mercedes", "audi", "bmw", "gle", "gla"]):
        prices = [random.randint(7000000, 7500000) - i*100000 for i in range(6)] # 70-75L range
    else:
        prices = [random.randint(800000, 900000) - i*10000 for i in range(6)] # 8-9L range
    # --- END FIX ---

    try:
        plt.figure(figsize=(10, 6))
        plt.bar(labels, prices, color='#2563eb') # Blue-600
        plt.title(f"Price History for {car_details.strip()}", color='white', fontsize=16)
        plt.ylabel("Price (₹)", color='white', fontsize=12) # <-- FIXED Currency
        plt.xlabel("Month", color='white', fontsize=12)
        
        # Style the plot for a dark theme
        ax = plt.gca()
        ax.set_facecolor('#1f2937') # gray-800
        plt.gcf().set_facecolor('#1f2937')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.spines['left'].set_color('#4b5563') # gray-600
        ax.spines['bottom'].set_color('#4b5563')

        # Save to a byte buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        buf.seek(0)
        
        # Encode as base64
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        
        return {
            "title": f"Price History for {car_details.strip()}",
            "image_data": image_base64
        }
        
    except Exception as e:
        print(f"Error generating graph: {e}")
        return {"title": "Error generating graph", "image_data": None}


# --- EXTERNAL API MOCKS / REAL CALLS (CURRENCY FIXED) ---

async def get_real_news(car_details: str) -> dict:
    """ REAL: Calls the NewsAPI. (Unchanged from your file) """
    if not NEWS_API_KEY:
        return {"intent": "get_news", "summary": "NewsAPI key not configured.", "articles": []}
        
    async with httpx.AsyncClient() as client:
        try:
            params = {"q": car_details, "apiKey": NEWS_API_KEY, "pageSize": 3, "sortBy": "relevancy"}
            response = await client.get("https://newsapi.org/v2/everything", params=params, timeout=10.0)
            response.raise_for_status() # Raise exception for 4xx/5xx
            articles = response.json().get("articles", [])
            return {
                "intent": "get_news",
                "summary": f"Found {len(articles)} relevant articles for {car_details}.",
                "articles": [{"title": a['title'], "source": a['source']['name'], "url": a['url']} for a in articles]
            }
        except Exception as e:
            print(f"NewsAPI call failed: {e}")
            return {"intent": "get_news", "summary": f"Could not fetch news: {e}", "articles": []}

async def real_get_reddit_sentiment(car_details: str) -> dict:
    """ REAL: Searches r/cars on Reddit and runs VADER sentiment analysis. (Unchanged from yourfile) """
    headers = {"User-Agent": "DealershipAI/1.0 by HackathonTeam"}
    params = { "q": car_details, "sort": "relevance", "t": "month", "limit": 10, "type": "link,self" }
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"https://www.reddit.com/r/cars/search.json", params=params, headers=headers, timeout=10.0)
            if response.status_code != 200:
                return {"summary": "Could not fetch Reddit data.", "sentiment_score": 0.5, "source": "Reddit (API Error)"}

            posts = response.json().get("data", {}).get("children", [])
            if not posts:
                return {"summary": "No recent posts found on r/cars.", "sentiment_score": 0.5, "source": "r/cars"}
            sentiment_scores = []
            top_post_title = "N/A"
            for i, post in enumerate(posts):
                title = post['data']['title']
                selftext = post['data']['selftext']
                if i == 0: top_post_title = title
                full_text = title + " " + selftext
                compound_score = vader_analyzer.polarity_scores(full_text)['compound']
                sentiment_scores.append(compound_score)
            if not sentiment_scores:
                return {"summary": "No text found in recent posts.", "sentiment_score": 0.5, "source": "r/cars"}

            avg_compound = sum(sentiment_scores) / len(sentiment_scores)
            normalized_score = (avg_compound + 1) / 2 # Normalize from [-1, 1] to [0, 1]
            summary = f"Analyzed {len(posts)} posts. Top post: '{top_post_title}'. Overall sentiment is {'positive' if avg_compound > 0.1 else 'negative' if avg_compound < -0.1 else 'neutral'}."
            return { "source": "r/cars (Live)", "sentiment_score": round(normalized_score, 2), "summary": summary }
        except Exception as e:
            print(f"Reddit analysis failed: {e}")
            return {"summary": f"Reddit analysis failed: {e}", "sentiment_score": 0.5, "source": "Reddit (Exception)"}

async def get_llm_synthesis(data_context: str) -> str:
    """
    Calls the OpenRouter LLM to synthesize a final report.
    """
    print("Running: get_llm_synthesis (OpenRouter)")
    if not OPENROUTER_API_KEY:
        return "LLM Synthesis is not configured. (API Key Missing)"
       
    synthesis_prompt = f"""
You are an expert sales strategist at an Indian car dealership.
I have run all the models and APIs. All prices are in Indian Rupees (₹).
Here is the raw data:

{data_context}

Please write a final, persuasive recommendation for me (the sales executive).
Start by addressing the car I asked about.
Be confident and direct.
Use **bold** for key numbers and *italic* for your reasoning.
Do not just list the data; interpret it for me and give me a final action plan.
Make sure all currency symbols are ₹.
"""
   
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "Hackathon Dealership AI"
    }
    body = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": synthesis_prompt},
        ],
        # We are NOT asking for JSON here, we want a text response
    }
   
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(OPENROUTER_API_URL, headers=headers, json=body, timeout=20.0)
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content']
            else:
                return f"Error: LLM synthesis failed ({response.status_code})."
    except Exception as e:
        return f"Error: LLM synthesis failed ({e})."
    

async def real_get_competitor_rates_from_search(car_details: str) -> dict:
    """
    NEW: Uses the existing Google Search API to find competitor prices.
    """
    print("Running: real_get_competitor_rates_from_search (Re-using Google Search)")
    # Create a very specific search query
    search_query = f'"{car_details}" price at local dealerships'
   
    # Call your existing web search function
    search_data = await get_real_web_search(search_query)
   
    competitors_list = []
    summary = "No specific prices found in snippets."

    if search_data.get("search_results"):
        summary = "Found prices in web snippets."
        for item in search_data["search_results"]:
            # We re-format the search results to fit the competitor card
            competitors_list.append({
                "dealer": item.get('source', 'Web Result'),
                "price": item.get('snippet', 'See link') # The snippet often contains the price
            })
           
    return {
        "intent": "get_competitor_pricing",
        "local_avg_price": "See Snippets", # We can't get an "avg" from snippets alone
        "competitors": competitors_list,
        "summary": summary
    }

async def get_real_web_search(search_query: str) -> dict:
    """ REAL: Calls the Google Custom Search API. (Unchanged from your file) """
    if not GOOGLE_SEARCH_API_KEY or not GOOGLE_SEARCH_CX_ID:
        return {"intent": "get_web_search", "summary": "Google Search API not configured.", "search_results": []}

    async with httpx.AsyncClient() as client:
        try:
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                "key": GOOGLE_SEARCH_API_KEY,
                "cx": GOOGLE_SEARCH_CX_ID,
                "q": search_query,
                "num": 3 # Get top 3 results
            }
            response = await client.get(url, params=params, timeout=10.0)
            response.raise_for_status()
            results = response.json().get("items", [])
            return {
                "intent": "get_web_search",
                "summary": f"Found {len(results)} web results for '{search_query}'.",
                "search_results": [
                    {"title": r['title'], "link": r['link'], "snippet": r['snippet'], "source": r['displayLink']}
                    for r in results
                ]
            }
        except Exception as e:
            print(f"Google Search API call failed: {e}")
            return {"intent": "get_web_search", "summary": f"Could not perform web search: {e}", "search_results": []}


# --- AI Agent Intent Classifier ---

async def get_llm_intent(query: str) -> dict:
    """ Calls the OpenRouter LLM to classify the user's intent. (Unchanged from your file) """
    if not OPENROUTER_API_KEY:
        print("OPENROUTER_API_KEY not set. Using simulation.")
        return simulate_intent(query)
        
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost", # Recommended by OpenRouter
        "X-Title": "Hackathon Dealership AI" # Recommended by OpenRouter
    }
    
    body = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": query}
        ],
        "response_format": {"type": "json_object"}
    }
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(OPENROUTER_API_URL, headers=headers, json=body, timeout=20.0)
            
            if response.status_code == 200:
                json_content_str = response.json()['choices'][0]['message']['content']
                return json.loads(json_content_str)
            else:
                print(f"Error from OpenRouter: {response.status_code} - {response.text}")
                return simulate_intent(query) # Fallback to simulation
                
    except Exception as e:
        print(f"Exception during LLM call: {e}")
        return simulate_intent(query) # Fallback to simulation

def simulate_intent(query: str) -> dict:
    """ Fallback simulation, now updated with new intents. (Unchanged from your file) """
    query_lower = query.lower()
    
    car_match = re.search(r"(for|on|about) the ([\w\s\d-]+)", query_lower)
    car_details = "Unknown Car"
    if car_match:
        car_details = car_match.group(2).strip()
    elif len(query_lower.split()) > 2: # simple guess
        car_details = query
        
    search_query = query

    if "news" in query_lower:
        return {"intent": "get_news", "car_details": car_details}
    if "sentiment" in query_lower or "reddit" in query_lower:
        return {"intent": "get_sentiment", "car_details": car_details}
    if "competitor" in query_lower or "competing" in query_lower:
        return {"intent": "get_competitor_pricing", "car_details": car_details}
    if "discount" in query_lower or "price" in query_lower:
        return {"intent": "get_discount", "car_details": car_details}
    if "demand" in query_lower or "selling well" in query_lower:
        return {"intent": "get_demand"}
    if "graph" in query_lower or "history" in query_lower:
        return {"intent": "get_graph", "car_details": car_details}
    if "what is" in query_lower or "search for" in query_lower or "look up" in query_lower:
        return {"intent": "get_web_search", "search_query": search_query}
    
    return {"intent": "general_report"}

# --- FastAPI App Initialization (Unchanged from your file) ---

app = FastAPI(
    title="Dealership AI Agent API",
    description="API for the hackathon project, routing queries to ML models and logging feedback."
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows all origins (for hackathon)
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods
    allow_headers=["*"], # Allows all headers
)

# --- API Endpoints (MERGED) ---

@app.get("/")
def read_root():
    return {"status": "Dealership AI Agent API is running."}

@app.post("/api/chat", response_model=ChatResponse)
async def chat_agent(request: ChatRequest):
    """
    This is your main "AI Agent" endpoint.
    It routes to internal models and external APIs.
    """
    query = request.query
    
    # --- 1. Get Intent from LLM ---
    llm_response = await get_llm_intent(query)
    intent = llm_response.get("intent", "general_report")
    car_details = llm_response.get("car_details", query)
    search_query = llm_response.get("search_query", query)
            
    print(f"Query: '{query}' -> Intent: '{intent}'")

    # --- 2. Tool/Model Calling (Based on LLM-derived intent) ---
    
    if intent == "get_discount":
        # --- "Full Report" intent: NOW CALLS THE REAL DEMAND MODEL ---
        
        # 1. Get base features from (mock) "Current Cars" DB
        car_features = get_car_features_from_csv_data(car_details) # <-- UPDATED (no await)
        
        # 2. Get specific demand score (REAL MODEL)
        specific_demand_score = ml_models.get_demand_score_for_car(car_features)
        
        # 3. Call external APIs (REAL)
        sentiment_data = await real_get_reddit_sentiment(car_details)
        competitor_data = await real_get_competitor_rates_from_search(car_details) # <-- This is the parallel Google Search
        
        # 4. Get price prediction (SMART MOCK)
        report_data = mock_get_discount_prediction(
            car_details, car_features, specific_demand_score, competitor_data
        )
        
        # 5. Combine all data into one report for the LLM
        report_data['real_demand_score'] = round(specific_demand_score, 2)
        report_data['external_sentiment'] = sentiment_data
        report_data['competitor_analysis'] = competitor_data
        
        # --- 6. NEW: Call LLM for Synthesis ---
        # Create a clean context string for the LLM
        data_context = f"""
        - Car: {report_data['car_identified']}
        - Base Price (from {report_data['data_source']}): {report_data['base_price']}
        - Days in Stock: {report_data['days_in_inventory']}
        - Competitor Price: {report_data['competitor_price_found']}
        - REAL Demand Score: {report_data['real_demand_score']} / 1.0 (Category: {report_data['demand_category']})
        - REAL Public Sentiment: {sentiment_data['sentiment_score']} / 1.0 (Summary: {sentiment_data['summary']})
        - Plausible Price Recommendation: {report_data['predicted_sale_price']} ({report_data['recommended_discount_pct']}% discount)
        - Competitor Snippets: {json.dumps(competitor_data['competitors'], indent=2)}
        """
        
        response_text = await get_llm_synthesis(data_context)

        return ChatResponse(response_text=response_text, report_data=report_data)

    elif intent == "get_demand":
        report_data = mock_get_demand_forecast()
        response_text = f"Here's the 30-day *general* market forecast: \n- **High Demand:** {', '.join(report_data['high_demand'])} \n- **Low Demand:** {', '.join(report_data['low_demand'])}."
        response_text += "\n\n(To get the demand for a *specific* car, ask for the discount report.)"
        return ChatResponse(response_text=response_text, report_data=report_data)

    elif intent == "get_graph":
        chart_data = mock_get_price_graph_data(car_details)
        if chart_data["image_data"]:
            response_text = f"Here is the price history graph for the {car_details} (Prices in ₹)."
        else:
            response_text = f"Sorry, I couldn't generate the price graph for {car_details}."
        return ChatResponse(response_text=response_text, chart_data=chart_data)

    elif intent == "get_news":
        report_data = await get_real_news(car_details)
        if report_data['articles']:
            response_text = f"Here's the recent news for the {car_details}:"
        else:
            response_text = f"No recent news found for {car_details}."
        return ChatResponse(response_text=response_text, report_data=report_data)

    elif intent == "get_sentiment":
        report_data = await real_get_reddit_sentiment(car_details)
        response_text = f"Here's the sentiment for the {car_details} from {report_data['source']}:\n"
        response_text += f"- **Score:** {report_data['sentiment_score']}/1.0\n"
        response_text += f"- **Summary:** {report_data['summary']}"
        return ChatResponse(response_text=response_text, report_data=report_data)

    elif intent == "get_competitor_pricing":
        report_data = await real_get_competitor_rates_from_search(car_details) # <-- FIXED
        response_text = f"Here's what I found for competitor pricing for the {car_details} (Prices in ₹):\n"
        response_text += f"- **Summary:** {report_data['summary']}\n- **Sources:**\n"
        response_text += "\n".join([f"  - {c['dealer']}: {c['price']}" for c in report_data['competitors']])
        return ChatResponse(response_text=response_text, report_data=report_data)

    elif intent == "get_web_search":
        report_data = await get_real_web_search(search_query)
        if report_data['search_results']:
            response_text = f"Here's what I found online for '{search_query}':"
        else:
            response_text = f"Sorry, I couldn't find any web results for '{search_query}'."
        return ChatResponse(response_text=response_text, report_data=report_data)

    else: # Catches "general_report"
        response_text = "I can help with discount recommendations, demand forecasts, or web searches. What would you like to know?"
        return ChatResponse(response_text=response_text)


@app.post("/api/feedback")
async def handle_feedback(request: FeedbackRequest):
    """
    NEW: This endpoint logs executive feedback to Firestore. (Unchanged from your file)
    """
    if not db:
        print("Firestore not initialized. Feedback not logged.")
        # Still return 200 so the client doesn't see an error
        return {"status": "ok", "message": "Feedback received (logging disabled)."}

    try:
        feedback_data = {
            "query": request.query,
            "feedback_reason": request.feedback_reason,
            "timestamp": datetime.now(firestore.firestore.SERVER_TIMESTAMP)
        }
        
        # Add a new doc with a generated ID to the 'feedback' collection
        update_time, doc_ref = db.collection('feedback').add(feedback_data)
        
        print(f"Feedback logged successfully: {doc_ref.id}")
        return {"status": "ok", "document_id": doc_ref.id}
        
    except Exception as e:
        print(f"Error logging feedback to Firestore: {e}")
        raise HTTPException(status_code=500, detail="Error logging feedback.")


# --- Main Entry Point ---
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

