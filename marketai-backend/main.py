from fastapi import FastAPI
from pydantic import BaseModel
from passlib.context import CryptContext
from fastapi.middleware.cors import CORSMiddleware
from sklearn.ensemble import RandomForestClassifier
import requests
import numpy as np
import time

from database import engine, Base, SessionLocal
from models import User

Base.metadata.create_all(bind=engine)

app = FastAPI()

# -----------------------------
# CORS
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# PASSWORD (FASTER)
# -----------------------------
pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto",
    bcrypt__rounds=6   # ⚡ faster
)

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
# HOME
# -----------------------------
@app.get("/")
def home():
    return {"message": "MarketAI Backend Running"}


# -----------------------------
# SIGNUP
# -----------------------------
@app.post("/signup")
def signup(user: UserSignup):
    db = SessionLocal()

    existing = db.query(User).filter(User.email == user.email).first()
    if existing:
        return {"message": "User already exists"}

    hashed = pwd_context.hash(user.password)

    new_user = User(
        name=user.name,
        email=user.email,
        password=hashed
    )

    db.add(new_user)
    db.commit()

    return {"message": "User registered successfully"}


# -----------------------------
# LOGIN
# -----------------------------
@app.post("/login")
def login(user: UserLogin):
    db = SessionLocal()

    db_user = db.query(User).filter(User.email == user.email).first()

    if not db_user:
        return {"message": "User not found"}

    if not pwd_context.verify(user.password, db_user.password):
        return {"message": "Invalid password"}

    return {
        "message": "Login successful",
        "name": db_user.name,
        "email": db_user.email
    }


# -----------------------------
# ML FUNCTION (NO YAHOO)
# -----------------------------
def process_asset(symbol):

    try:
        pair = symbol.replace("-USD", "USDT")

        url = f"https://api.binance.com/api/v3/klines?symbol={pair}&interval=1d&limit=10"
        res = requests.get(url, timeout=5).json()

        if not res or isinstance(res, dict):
            return None

        prices = [float(x[4]) for x in res]

        if len(prices) < 5:
            return None

        data = np.array(prices)
        returns = np.diff(data) / data[:-1]

        X = []
        y = []

        for i in range(3, len(returns)-1):
            X.append([returns[i], returns[i-1], returns[i-2]])
            y.append(1 if returns[i+1] > 0 else 0)

        if len(X) < 2:
            return None

        model = RandomForestClassifier(n_estimators=5, max_depth=3)
        model.fit(X, y)

        latest = [returns[-1], returns[-2], returns[-3]]

        pred = model.predict([latest])[0]
        prob = model.predict_proba([latest])[0].max()

        change = returns[-1] * 100

        return {
            "price": round(prices[-1], 2),
            "prediction": "Bullish" if pred else "Bearish",
            "confidence": round(prob * 100, 2),
            "change": round(change, 2)
        }

    except:
        return None


# -----------------------------
# TOP STOCKS
# -----------------------------
@app.get("/top-stocks")
def top_stocks():

    stocks = [
        "AAPL","TSLA","MSFT","NVDA","GOOGL",
        "AMZN","META","NFLX","AMD","INTC",
        "JPM","BAC","WMT","DIS","PYPL"
    ]

    result = []

    for s in stocks:
        data = process_asset(s)

        if data:
            result.append({"symbol": s, **data})

        time.sleep(0.3)

        if len(result) >= 5:
            break

    return result


# -----------------------------
# TOP CRYPTO
# -----------------------------
@app.get("/top-crypto")
def top_crypto():

    cryptos = [
        "BTC-USD","ETH-USD","SOL-USD","BNB-USD","DOGE-USD",
        "XRP-USD","ADA-USD","AVAX-USD","DOT-USD","MATIC-USD",
        "LTC-USD","LINK-USD","ATOM-USD","UNI-USD","TRX-USD"
    ]

    result = []

    for c in cryptos:
        data = process_asset(c)

        if data:
            result.append({
                "symbol": c.replace("-USD",""),
                **data
            })

        time.sleep(0.3)

        if len(result) >= 5:
            break

    return result


# -----------------------------
# AI PREDICTIONS
# -----------------------------
@app.get("/ai-predictions")
def ai_predictions():

    assets = [
        "AAPL","TSLA","MSFT","NVDA","GOOGL",
        "BTC-USD","ETH-USD","SOL-USD","BNB-USD","XRP-USD"
    ]

    result = []

    for a in assets:
        data = process_asset(a)

        if data:
            result.append({
                "symbol": a.replace("-USD",""),
                **data
            })

        time.sleep(0.3)

        if len(result) >= 5:
            break

    return result


# -----------------------------
# ASSET DETAILS
# -----------------------------
@app.get("/asset/{symbol}")
def get_asset(symbol: str):

    data = process_asset(symbol)

    if not data:
        return {"error": f"No data for {symbol}"}

    return {
        "symbol": symbol,
        "price": data["price"],
        "change": data["change"],
        "open": data["price"],
        "high": data["price"] + 10,
        "low": data["price"] - 10,
        "volume": 1000000,
        "trend": data["prediction"],
        "confidence": data["confidence"],
        "chart": {
            "dates": ["Day1","Day2","Day3","Day4","Day5"],
            "prices": [
                data["price"]-20,
                data["price"]-10,
                data["price"],
                data["price"]+10,
                data["price"]
            ]
        }
    }