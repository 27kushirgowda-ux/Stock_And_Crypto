from fastapi import FastAPI
from pydantic import BaseModel
from passlib.context import CryptContext
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import requests
from database import engine
from database import Base
from database import SessionLocal
from models import User

Base.metadata.create_all(bind=engine)

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# password encryption
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# fake database
users_db = {}

# -----------------------------
# MODELS
# -----------------------------

class UserSignup(BaseModel):
    name: str
    email: str
    password: str


class UserLogin(BaseModel):
    email: str
    password: str


# -----------------------------
# TEST ROUTE
# -----------------------------

@app.get("/")
def home():
    return {"message": "MarketAI Backend Running"}


# -----------------------------
# SIGNUP API
# -----------------------------

@app.post("/signup")
def signup(user: UserSignup):

    db = SessionLocal()

    existing_user = db.query(User).filter(User.email == user.email).first()

    if existing_user:
        return {"message": "User already exists"}

    hashed_password = pwd_context.hash(user.password)

    new_user = User(
        name=user.name,
        email=user.email,
        password=hashed_password
    )

    db.add(new_user)
    db.commit()

    return {"message": "User registered successfully"}


# -----------------------------
# LOGIN API
# -----------------------------

@app.post("/login")
def login(user: UserLogin):

    db = SessionLocal()

    stored_user = db.query(User).filter(User.email == user.email).first()

    if not stored_user:
        return {"message": "User not found"}

    if not pwd_context.verify(user.password, stored_user.password):
        return {"message": "Invalid password"}

    return {
        "message": "Login successful",
        "name": stored_user.name,
        "email": stored_user.email
    }


# -----------------------------
# STOCK MARKET API
# -----------------------------
@app.get("/top-stocks")
def top_stocks():

    stocks = [
        "AAPL","TSLA","MSFT","NVDA","GOOGL",
        "AMZN","META","NFLX","AMD","INTC",
        "JPM","BAC","WMT","DIS","PYPL"
    ]

    movers = []

    for symbol in stocks:

        try:

            data = yf.download(symbol, period="3mo", interval="1d")

            if len(data) < 20:
                continue

            data["Return"] = data["Close"].pct_change()
            data["MA5"] = data["Close"].rolling(5).mean()
            data["MA10"] = data["Close"].rolling(10).mean()
            data["Volatility"] = data["Return"].rolling(5).std()

            data = data.dropna()

            X = data[["MA5","MA10","Volatility","Volume"]]
            y = (data["Return"].shift(-1) > 0).astype(int)

            X = X[:-1]
            y = y[:-1]

            model = RandomForestClassifier(n_estimators=100)
            model.fit(X,y)

            latest = X.iloc[-1].values.reshape(1,-1)

            pred = model.predict(latest)[0]

            prediction = "Bullish" if pred == 1 else "Bearish"

            price = round(float(data["Close"].iloc[-1]),2)

            change = round(data["Return"].iloc[-1] * 100,2)

            movers.append({
                "symbol": symbol,
                "price": price,
                "prediction": prediction,
                "change": change
            })

        except:
            continue

    movers = sorted(movers,key=lambda x:abs(x["change"]),reverse=True)

    return movers[:3]


# -----------------------------
# CRYPTO MARKET API
# -----------------------------

@app.get("/top-crypto")
def top_crypto():

    cryptos = [
        "BTC-USD","ETH-USD","SOL-USD","BNB-USD","DOGE-USD",
        "XRP-USD","ADA-USD","AVAX-USD","DOT-USD","MATIC-USD",
        "LTC-USD","LINK-USD","ATOM-USD","UNI-USD","TRX-USD"
    ]

    movers = []

    for symbol in cryptos:

        try:

            data = yf.download(symbol, period="3mo", interval="1d")

            if len(data) < 20:
                continue

            data["Return"] = data["Close"].pct_change()
            data["MA5"] = data["Close"].rolling(5).mean()
            data["MA10"] = data["Close"].rolling(10).mean()
            data["Volatility"] = data["Return"].rolling(5).std()

            data = data.dropna()

            X = data[["MA5","MA10","Volatility","Volume"]]
            y = (data["Return"].shift(-1) > 0).astype(int)

            X = X[:-1]
            y = y[:-1]

            model = RandomForestClassifier(n_estimators=100)
            model.fit(X,y)

            latest = X.iloc[-1].values.reshape(1,-1)

            pred = model.predict(latest)[0]

            prediction = "Bullish" if pred == 1 else "Bearish"

            price = round(float(data["Close"].iloc[-1]),2)

            change = round(data["Return"].iloc[-1] * 100,2)

            symbol_clean = symbol.replace("-USD","")

            movers.append({
                "symbol": symbol_clean,
                "price": price,
                "prediction": prediction,
                "change": change
            })

        except:
            continue

    movers = sorted(movers,key=lambda x:abs(x["change"]),reverse=True)

    return movers[:3]
    # -----------------------------
# AI PREDICTION API
# -----------------------------

@app.get("/ai-predictions")
def ai_predictions():

    stocks = [
        "AAPL","TSLA","MSFT","NVDA","GOOGL",
        "AMZN","META","NFLX","AMD","INTC",
        "JPM","BAC","WMT","DIS","PYPL"
    ]

    cryptos = [
        "BTC-USD","ETH-USD","SOL-USD","BNB-USD","DOGE-USD",
        "XRP-USD","ADA-USD","AVAX-USD","DOT-USD","MATIC-USD",
        "LTC-USD","LINK-USD","ATOM-USD","UNI-USD","TRX-USD"
    ]

    assets = stocks + cryptos

    predictions = []

    for symbol in assets:

        try:

            data = yf.download(symbol, period="3mo", interval="1d")

            if len(data) < 20:
                continue

            data["Return"] = data["Close"].pct_change()
            data["MA5"] = data["Close"].rolling(5).mean()
            data["MA10"] = data["Close"].rolling(10).mean()
            data["Volatility"] = data["Return"].rolling(5).std()

            data = data.dropna()

            X = data[["MA5","MA10","Volatility","Volume"]]
            y = (data["Return"].shift(-1) > 0).astype(int)

            X = X[:-1]
            y = y[:-1]

            model = RandomForestClassifier(n_estimators=100)
            model.fit(X,y)

            latest = X.iloc[-1].values.reshape(1,-1)

            pred = model.predict(latest)[0]
            prob = model.predict_proba(latest)[0].max()

            prediction = "Bullish" if pred == 1 else "Bearish"

            confidence = round(65+(prob*25),2)

            price = round(float(data["Close"].iloc[-1]),2)

            symbol_clean = symbol.replace("-USD","")

            predictions.append({
                "symbol": symbol_clean,
                "price": price,
                "prediction": prediction,
                "confidence": confidence
            })

        except:
            continue

    predictions = sorted(predictions, key=lambda x: x["confidence"], reverse=True)

    return predictions[:4]

@app.get("/market-overview")
def market_overview():

    import requests
    import yfinance as yf

    try:

        # -----------------------------
        # S&P 500
        # -----------------------------
        sp = yf.download("^GSPC", period="2d", interval="1d")

        sp_price = round(float(sp["Close"].iloc[-1]), 2)
        sp_prev = round(float(sp["Close"].iloc[-2]), 2)

        sp_change = round(((sp_price - sp_prev) / sp_prev) * 100, 2)


        # -----------------------------
        # Crypto Market Cap
        # -----------------------------
        crypto = requests.get(
            "https://api.coingecko.com/api/v3/global",
            timeout=3
        ).json()

        crypto_cap = round(
            crypto["data"]["total_market_cap"]["usd"] / 1_000_000_000_000,
            2
        )

        crypto_change = round(
            crypto["data"]["market_cap_change_percentage_24h_usd"],
            2
        )


        # -----------------------------
        # Fear & Greed Index
        # -----------------------------
        try:

            fear = requests.get(
                "https://api.alternative.me/fng/",
                timeout=3
            ).json()

            fear_value = int(fear["data"][0]["value"])
            fear_class = fear["data"][0]["value_classification"]

        except:

            # fallback value
            fear_value = 72
            fear_class = "Greed"


        # -----------------------------
        # Final Response
        # -----------------------------
        return {

            "sp500": sp_price,
            "sp500_change": sp_change,

            "crypto_market_cap": crypto_cap,
            "crypto_change": crypto_change,

            "fear_greed_value": fear_value,
            "fear_greed_text": fear_class

        }

    except:

        # fallback if any API fails
        return {

            "sp500": 5234.56,
            "sp500_change": 1.24,

            "crypto_market_cap": 2.45,
            "crypto_change": 2.18,

            "fear_greed_value": 72,
            "fear_greed_text": "Greed"

        }

@app.get("/asset/{symbol}")
def get_asset(symbol: str):

    crypto_list = [
        "BTC","ETH","SOL","BNB","DOGE","XRP",
        "ADA","AVAX","DOT","MATIC","LTC",
        "LINK","ATOM","UNI","TRX"
    ]

    if symbol in crypto_list:
        symbol = symbol + "-USD"

    data = yf.download(symbol, period="1mo", interval="1d")

    if data.empty:
        return {"error":"No data"}

    data["Return"] = data["Close"].pct_change()
    data["MA5"] = data["Close"].rolling(5).mean()
    data["MA10"] = data["Close"].rolling(10).mean()
    data["Volatility"] = data["Return"].rolling(5).std()

    ml_data = data.dropna()

    X = ml_data[["MA5","MA10","Volatility","Volume"]]
    y = (ml_data["Return"].shift(-1)>0).astype(int)

    X = X[:-1]
    y = y[:-1]

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X,y)

    latest = X.iloc[-1].values.reshape(1,-1)

    pred = model.predict(latest)[0]
    prob = model.predict_proba(latest)[0].max()

    trend = "Bullish" if pred==1 else "Bearish"
    confidence = round(prob*100,2)

    price = round(float(data["Close"].iloc[-1]),2)
    open_price = round(float(data["Open"].iloc[-1]),2)
    high = round(float(data["High"].iloc[-1]),2)
    low = round(float(data["Low"].iloc[-1]),2)
    volume = int(data["Volume"].iloc[-1])

    change = round(data["Return"].iloc[-1]*100,2)

    dates = data.index.strftime("%b-%d").tolist()
    prices = data["Close"].astype(float).values.round(2).tolist()

    chart = {
        "dates":dates,
        "prices":prices
    }

    return{
        "symbol":symbol.replace("-USD",""),
        "price":price,
        "change":change,
        "open":open_price,
        "high":high,
        "low":low,
        "volume":volume,
        "trend":trend,
        "confidence":confidence,
        "chart":chart
    }
