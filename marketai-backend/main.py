from fastapi import FastAPI
from pydantic import BaseModel
from passlib.context import CryptContext
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
import requests

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
# PASSWORD (FAST)
# -----------------------------
pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto",
    bcrypt__rounds=10
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
# COMMON ML FUNCTION
# -----------------------------
def process_asset(symbol):

    data = yf.download(
        symbol,
        period="3mo",
        interval="1d",
        progress=False,
        threads=False
    )

    if data.empty or len(data) < 20:
        return None

    data["Return"] = data["Close"].pct_change()
    data["MA5"] = data["Close"].rolling(5).mean()
    data["MA10"] = data["Close"].rolling(10).mean()
    data["Volatility"] = data["Return"].rolling(5).std()

    data = data.dropna()

    X = data[["MA5","MA10","Volatility","Volume"]]
    y = (data["Return"].shift(-1) > 0).astype(int)

    X, y = X[:-1], y[:-1]

    model = RandomForestClassifier(n_estimators=10, max_depth=5)
    model.fit(X,y)

    latest = X.iloc[-1].values.reshape(1,-1)

    pred = model.predict(latest)[0]
    prob = model.predict_proba(latest)[0].max()

    return {
        "price": round(float(data["Close"].iloc[-1]),2),
        "prediction": "Bullish" if pred else "Bearish",
        "confidence": round(prob*100,2),
        "change": round(data["Return"].iloc[-1]*100,2)
    }


# -----------------------------
# TOP STOCKS (15)
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
        try:
            data = process_asset(s)
            if data:
                result.append({
                    "symbol": s,
                    **data
                })
        except:
            continue

    return sorted(result, key=lambda x: abs(x["change"]), reverse=True)[:5]


# -----------------------------
# TOP CRYPTO (15)
# -----------------------------
@app.get("/top-crypto")
def top_crypto():

    cryptos = [
        "BTC-USD","ETH-USD","SOL-USD","BNB-USD","DOGE-USD",
        "XRP-USD","ADA-USD","AVAX-USD","DOT-USD",
        "LTC-USD","LINK-USD","ATOM-USD","TRX-USD"
    ]

    result = []

    for c in cryptos:
        try:
            data = process_asset(c)
            if data:
                result.append({
                    "symbol": c.replace("-USD",""),
                    **data
                })
        except:
            continue

    return sorted(result, key=lambda x: abs(x["change"]), reverse=True)[:5]


# -----------------------------
# AI PREDICTIONS
# -----------------------------
@app.get("/ai-predictions")
def ai_predictions():

    assets = [
        "AAPL","TSLA","MSFT","NVDA","GOOGL",
        "AMZN","META","NFLX","AMD","INTC",
        "JPM","BAC","WMT","DIS","PYPL",
        "BTC-USD","ETH-USD","SOL-USD","BNB-USD","DOGE-USD",
        "XRP-USD","ADA-USD","AVAX-USD","DOT-USD",
        "LTC-USD","LINK-USD","ATOM-USD","TRX-USD"
    ]

    result = []

    for a in assets:
        try:
            data = process_asset(a)
            if data:
                result.append({
                    "symbol": a.replace("-USD",""),
                    **data
                })
        except:
            continue

    return sorted(result, key=lambda x: x["confidence"], reverse=True)[:5]


# -----------------------------
# MARKET OVERVIEW
# -----------------------------
@app.get("/market-overview")
def market_overview():

    sp = yf.download("^GSPC", period="2d", interval="1d", progress=False)

    sp_price = round(float(sp["Close"].iloc[-1]), 2)
    sp_prev = round(float(sp["Close"].iloc[-2]), 2)

    sp_change = round(((sp_price - sp_prev) / sp_prev) * 100, 2)

    crypto = requests.get("https://api.coingecko.com/api/v3/global").json()

    return {
        "sp500": sp_price,
        "sp500_change": sp_change,
        "crypto_market_cap": round(
            crypto["data"]["total_market_cap"]["usd"] / 1_000_000_000_000, 2
        ),
        "crypto_change": round(
            crypto["data"]["market_cap_change_percentage_24h_usd"], 2
        ),
        "fear_greed_value": 72,
        "fear_greed_text": "Greed"
    }