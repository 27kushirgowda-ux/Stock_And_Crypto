from fastapi import FastAPI
from pydantic import BaseModel
from passlib.context import CryptContext
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
import requests
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
    bcrypt__rounds=8   # faster login/signup
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
# SIGNUP (FAST)
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
    db.close()

    return {"message": "User registered successfully"}


# -----------------------------
# LOGIN (FAST)
# -----------------------------
@app.post("/login")
def login(user: UserLogin):
    db = SessionLocal()

    db_user = db.query(User).filter(User.email == user.email).first()

    if not db_user:
        return {"message": "User not found"}

    if not pwd_context.verify(user.password, db_user.password):
        return {"message": "Invalid password"}

    db.close()

    return {
        "message": "Login successful",
        "name": db_user.name,
        "email": db_user.email
    }


# -----------------------------
# ML FUNCTION (FAST + SAFE)
# -----------------------------
def process_asset(symbol):
    try:
        data = yf.download(
            symbol,
            period="1mo",
            interval="1d",
            progress=False,
            threads=False
        )

        if data.empty or len(data) < 10:
            return None

        data["Return"] = data["Close"].pct_change()
        data["MA5"] = data["Close"].rolling(5).mean()
        data["MA10"] = data["Close"].rolling(10).mean()
        data["Volatility"] = data["Return"].rolling(5).std()

        data = data.dropna()

        if len(data) < 5:
            return None

        X = data[["MA5","MA10","Volatility","Volume"]]
        y = (data["Return"].shift(-1) > 0).astype(int)

        X, y = X[:-1], y[:-1]

        model = RandomForestClassifier(n_estimators=5, max_depth=3)
        model.fit(X, y)

        latest = X.iloc[-1].values.reshape(1,-1)

        pred = model.predict(latest)[0]
        prob = model.predict_proba(latest)[0].max()

        return {
            "price": round(float(data["Close"].iloc[-1]),2),
            "prediction": "Bullish" if pred else "Bearish",
            "confidence": round(float(prob)*100,2),
            "change": round(float(data["Return"].iloc[-1])*100,2)
        }

    except Exception as e:
        print("ERROR:", symbol, e)
        return None


# -----------------------------
# TOP STOCKS (FAST)
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

            time.sleep(0.5)  # small delay to prevent API overload

        if len(result) >= 5:   # 🚀 SPEED CONTROL
            break

    return result


# -----------------------------
# TOP CRYPTO (FAST)
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

            time.sleep(0.5)  # small delay to prevent API overload

        if len(result) >= 5:
            break

    return result


# -----------------------------
# AI PREDICTIONS (FAST)
# -----------------------------
@app.get("/ai-predictions")
def ai_predictions():

    assets = [
        "AAPL","TSLA","MSFT","NVDA","GOOGL",
        "AMZN","META","NFLX","AMD","INTC",
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

            time.sleep(0.5)  # small delay to prevent API overload

        if len(result) >= 5:
            break

    return result


# -----------------------------
# MARKET OVERVIEW
# -----------------------------
@app.get("/market-overview")
def market_overview():

    try:
        sp = yf.download("^GSPC", period="5d", interval="1d", progress=False)

        close = sp["Close"]

        sp_price = float(close.iloc[-1])
        sp_prev = float(close.iloc[-2])

        sp_price = round(sp_price, 2)
        sp_change = round(((sp_price - sp_prev) / sp_prev) * 100, 2)

        crypto = requests.get(
            "https://api.coingecko.com/api/v3/global",
            timeout=5
        ).json()

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

    except Exception as e:
        return {"error": str(e)}


# -----------------------------
# ASSET DETAILS (MAIN FEATURE)
# -----------------------------
@app.get("/asset/{symbol}")
def get_asset(symbol: str):

    symbol = symbol.upper()

    crypto_list = [
        "BTC","ETH","SOL","BNB","DOGE","XRP",
        "ADA","AVAX","DOT","MATIC","LTC",
        "LINK","ATOM","UNI","TRX"
    ]

    yf_symbol = symbol + "-USD" if symbol in crypto_list else symbol

    data = yf.download(
        yf_symbol,
        period="1mo",
        interval="1d",
        progress=False,
        threads=False
    )

    if data.empty or len(data) < 10:
        return {"error": f"No data for {symbol}"}

    data["Return"] = data["Close"].pct_change()
    data["MA5"] = data["Close"].rolling(5).mean()
    data["MA10"] = data["Close"].rolling(10).mean()
    data["Volatility"] = data["Return"].rolling(5).std()

    data = data.dropna()

    X = data[["MA5","MA10","Volatility","Volume"]]
    y = (data["Return"].shift(-1) > 0).astype(int)

    X, y = X[:-1], y[:-1]

    model = RandomForestClassifier(n_estimators=5, max_depth=3)
    model.fit(X, y)

    latest = X.iloc[-1].values.reshape(1,-1)

    pred = model.predict(latest)[0]
    prob = model.predict_proba(latest)[0].max()

    return {
        "symbol": symbol,
        "price": round(float(data["Close"].iloc[-1]),2),
        "change": round(float(data["Return"].iloc[-1])*100,2),
        "open": round(float(data["Open"].iloc[-1]),2),
        "high": round(float(data["High"].iloc[-1]),2),
        "low": round(float(data["Low"].iloc[-1]),2),
        "volume": int(data["Volume"].iloc[-1]),
        "trend": "Bullish" if pred else "Bearish",
        "confidence": round(float(prob)*100,2),
        "chart": {
            "dates": data.index.strftime("%b-%d").tolist(),
            "prices": data["Close"].astype(float).tolist()
        }
    }

@app.get("/test-yahoo")
def test_yahoo():
    import yfinance as yf

    data = yf.download("AAPL", period="5d", interval="1d")

    if data.empty:
        return {"status": "FAILED", "reason": "Yahoo not working"}

    return {
        "status": "SUCCESS",
        "rows": len(data),
        "price": float(data["Close"].iloc[-1])
    }