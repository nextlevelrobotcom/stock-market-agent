import os
import warnings
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import pandas_ta as ta
from statsmodels.tsa.arima.model import ARIMA
import requests
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage
import json  # For parsing potential JSON output from LLM

# Import custom handler
from stock_data_ingestion import InfluxDBHandler

# Import Slack client
from slack_sdk import WebClient

# Suppress warnings (e.g., from ARIMA, pandas_ta, langchain)
warnings.filterwarnings("ignore")

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
SYMBOLS = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA", "META", "NFLX"]  # Stocks to analyze
DATA_INGESTION_DAYS = 2  # Fetch data for the last N days to ensure freshness
ANALYSIS_WINDOW_DAYS = 365  # Use last N days of data for analysis
ARIMA_ORDER = (5, 1, 0)  # Example (p, d, q) order for ARIMA - NEEDS TUNING!
FORECAST_CHANGE_THRESHOLD = 0.005  # 0.5% change threshold for quant recommendation
RECOMMENDATION_SCORE_THRESHOLD = 0.5  # Threshold for Buy/Sell in combined score

# LLM Configuration for Sentiment Analysis
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openrouter")  # 'openrouter' or 'ollama'
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
LLM_MODEL_NAME = os.getenv(
    "LLM_MODEL_NAME", "openai/gpt-3.5-turbo"
)  # Default model for OpenRouter or Ollama
LLM_TEMPERATURE = 0.1  # Low temperature for consistent scoring
NEWS_SENTIMENT_DAYS = 1  # Analyze news from the last N days
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Slack configuration
SLACK_TOKEN = os.getenv("SLACK_TOKEN")
SLACK_CHANNEL = os.getenv("SLACK_CHANNEL", "#test")  # Default channel if not set
SLACK_BOT_NAME = os.getenv(
    "SLACK_BOT_NAME", "Investment Consultant"
)  # Default channel if not set

# --- Helper Functions ---


def calculate_technical_indicators(stock_data_df, symbols):
    """Calculates technical indicators using pandas_ta."""
    tech_data = {}
    print("Calculating technical indicators...")
    if stock_data_df.empty:
        print("Stock data is empty. Skipping indicator calculation.")
        return tech_data

    for symbol in symbols:
        if symbol not in stock_data_df.columns:
            print(f"Symbol {symbol} not found in retrieved stock data. Skipping.")
            continue

        print(f"  Calculating for {symbol}...")
        # Create a copy and rename 'close' column if needed (assuming retrieve_data provides 'close')
        # NOTE: For full TA, retrieve_data should provide OHLCV columns.
        #       pandas_ta works best with columns named 'open', 'high', 'low', 'close', 'volume'.
        #       We proceed with 'close' as per the notebook's example.
        df = stock_data_df[[symbol]].copy()
        df.rename(columns={symbol: "close"}, inplace=True)

        # Check if 'close' column exists and is not empty
        if "close" in df.columns and not df["close"].isnull().all():
            try:
                # Calculate SMAs
                df.ta.sma(length=50, append=True)  # SMA_50
                df.ta.sma(length=200, append=True)  # SMA_200
                # Calculate EMA
                df.ta.ema(length=20, append=True)  # EMA_20
                # Calculate RSI
                df.ta.rsi(length=14, append=True)  # RSI_14
                # Calculate MACD
                df.ta.macd(
                    fast=12, slow=26, signal=9, append=True
                )  # MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9
                # Calculate Bollinger Bands
                df.ta.bbands(
                    length=20, std=2, append=True
                )  # BBL_20_2.0, BBM_20_2.0, BBU_20_2.0, BBB_20_2.0, BBP_20_2.0

                tech_data[symbol] = df
                print(f"  Indicators calculated for {symbol}.")
            except Exception as e:
                print(f"  Error calculating indicators for {symbol}: {e}")
        else:
            print(
                f"  'close' column missing or empty for {symbol}. Skipping indicator calculation."
            )

    return tech_data


def get_technical_recommendation(tech_data_dict, symbols):
    """Generates technical recommendations based on indicator signals."""
    tech_recommendations = {}
    final_signal_scores = {}
    print("Generating technical recommendations...")

    if not tech_data_dict:
        print("Technical data dictionary is empty. Skipping recommendations.")
        return {symbol: "No Data" for symbol in symbols}

    for symbol in symbols:
        if symbol not in tech_data_dict:
            tech_recommendations[symbol] = "No Data"
            print(f"  No technical data for {symbol}. Skipping.")
            continue

        df = tech_data_dict[symbol]
        print(f"  Generating for {symbol}...")

        # Drop rows with NaNs
        df_clean = df.dropna()

        if df_clean.empty:
            print(f"  Not enough data for {symbol} after dropping NaNs. Skipping.")
            tech_recommendations[symbol] = "Not Enough Data"
            continue

        # Get the latest data point
        latest = df_clean.iloc[-1]
        signals = {
            "RSI": 0,
            "MACD": 0,
            "MA": 0,
            "BB": 0,
        }  # 1 for Buy, -1 for Sell, 0 for Hold

        # --- Apply Rules (adapted from notebook) ---
        # RSI
        if "RSI_14" in latest and not pd.isna(latest["RSI_14"]):
            if latest["RSI_14"] < 30:
                signals["RSI"] = 1
            elif latest["RSI_14"] > 70:
                signals["RSI"] = -1

        # MACD Crossover (needs previous point)
        if len(df_clean) > 1:
            prev = df_clean.iloc[-2]
            if all(
                k in latest and not pd.isna(latest[k])
                for k in ["MACD_12_26_9", "MACDs_12_26_9"]
            ) and all(
                k in prev and not pd.isna(prev[k])
                for k in ["MACD_12_26_9", "MACDs_12_26_9"]
            ):
                if (
                    prev["MACD_12_26_9"] < prev["MACDs_12_26_9"]
                    and latest["MACD_12_26_9"] > latest["MACDs_12_26_9"]
                ):
                    signals["MACD"] = 1  # Bullish Crossover
                elif (
                    prev["MACD_12_26_9"] > prev["MACDs_12_26_9"]
                    and latest["MACD_12_26_9"] < latest["MACDs_12_26_9"]
                ):
                    signals["MACD"] = -1  # Bearish Crossover

        # Moving Averages
        if all(
            k in latest and not pd.isna(latest[k])
            for k in ["SMA_50", "SMA_200", "close"]
        ):
            if (
                latest["SMA_50"] > latest["SMA_200"]
                and latest["close"] > latest["SMA_50"]
            ):
                signals["MA"] = 1  # Golden Cross tendency
            elif (
                latest["SMA_50"] < latest["SMA_200"]
                and latest["close"] < latest["SMA_50"]
            ):
                signals["MA"] = -1  # Death Cross tendency

        # Bollinger Bands
        if all(
            k in latest and not pd.isna(latest[k])
            for k in ["BBL_20_2.0", "BBU_20_2.0", "close"]
        ):
            if latest["close"] <= latest["BBL_20_2.0"]:
                signals["BB"] = 1
            elif latest["close"] >= latest["BBU_20_2.0"]:
                signals["BB"] = -1

        # --- Combine Signals ---
        valid_signals = [s for s in signals.values()]  # Assuming 0 is valid (Hold)
        final_signal_score = (
            sum(valid_signals) / len(valid_signals) if valid_signals else 0
        )

        if final_signal_score >= RECOMMENDATION_SCORE_THRESHOLD:
            recommendation = "Buy"
        elif final_signal_score <= -RECOMMENDATION_SCORE_THRESHOLD:
            recommendation = "Sell"
        else:
            recommendation = "Hold"

        tech_recommendations[symbol] = recommendation
        print(
            f"  {symbol} - Signals: {signals}, Score: {final_signal_score:.2f}, Recommendation: {recommendation}"
        )
        final_signal_scores[symbol] = final_signal_score

    return tech_recommendations, final_signal_scores


def get_quantitative_recommendation(stock_data_df, symbols, arima_order, threshold):
    """Generates quantitative recommendations using ARIMA forecast."""
    quant_recommendations = {}
    forecast_ratios = {}
    print("Generating quantitative recommendations (ARIMA)...")

    if stock_data_df.empty:
        print("Stock data is empty. Skipping ARIMA modeling.")
        return {symbol: "No Data" for symbol in symbols}

    for symbol in symbols:
        if symbol not in stock_data_df.columns:
            print(f"  Symbol {symbol} not found in stock data. Skipping.")
            quant_recommendations[symbol] = "No Data"
            continue

        print(f"  Fitting ARIMA for {symbol}...")
        series = stock_data_df[symbol].dropna()

        if len(series) < (
            arima_order[0] + arima_order[2] + 10
        ):  # Basic check for enough data
            print(
                f"  Not enough data points for {symbol} to fit ARIMA({arima_order}). Skipping."
            )
            quant_recommendations[symbol] = "Not Enough Data"
            continue

        try:
            # Fit ARIMA model
            model = ARIMA(
                series,
                order=arima_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            model_fit = model.fit()
            # Forecast next step
            forecast_result = model_fit.get_forecast(steps=1)
            forecast_value = forecast_result.predicted_mean.iloc[0]

            # Generate recommendation based on forecast vs latest price
            latest_actual_price = series.iloc[-1]
            recommendation = "Hold"
            if forecast_value > latest_actual_price * (1 + threshold):
                recommendation = "Buy"
            elif forecast_value < latest_actual_price * (1 - threshold):
                recommendation = "Sell"

            if latest_actual_price > 0:
                forecast_ratios[symbol] = forecast_value / latest_actual_price
            else:
                forecast_ratios[symbol] = 0

            quant_recommendations[symbol] = recommendation
            print(
                f"  {symbol} - Latest: {latest_actual_price:.2f}, Forecast: {forecast_value:.2f}, Recommendation: {recommendation}"
            )

        except Exception as e:
            print(f"  Error fitting ARIMA for {symbol}: {e}")
            quant_recommendations[symbol] = "Error"  # Indicate failure

    return quant_recommendations, forecast_ratios


def combine_recommendations(tech_recs, quant_recs, symbols):
    """Combines technical and quantitative recommendations."""
    final_recommendations = {}
    print("Combining recommendations...")
    score_map = {
        "Buy": 1,
        "Hold": 0,
        "Sell": -1,
        "Not Enough Data": 0,
        "No Forecast": 0,
        "No Data": 0,
        "Error": 0,
    }

    for symbol in symbols:
        tech_rec = tech_recs.get(symbol, "No Data")
        quant_rec = quant_recs.get(symbol, "No Data")

        tech_score = score_map.get(tech_rec, 0)
        quant_score = score_map.get(quant_rec, 0)

        # Combine scores (simple average, ignoring non-actionable recommendations)
        valid_scores = []
        if tech_rec not in ["Not Enough Data", "No Data", "Error"]:
            valid_scores.append(tech_score)
        if quant_rec not in ["Not Enough Data", "No Forecast", "No Data", "Error"]:
            valid_scores.append(quant_score)

        combined_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0

        # Determine final recommendation
        final_rec = "Hold"
        if combined_score >= RECOMMENDATION_SCORE_THRESHOLD:
            final_rec = "Buy"
        elif combined_score <= -RECOMMENDATION_SCORE_THRESHOLD:
            final_rec = "Sell"

        final_recommendations[symbol] = {
            "Technical": tech_rec,
            "Quantitative": quant_rec,
            "Combined": final_rec,
            "Score": round(combined_score, 2),
        }
        print(
            f"  {symbol} - Tech: {tech_rec}, Quant: {quant_rec}, Combined: {final_rec} (Score: {combined_score:.2f})"
        )

    return final_recommendations


def get_news_sentiment_scores(
    news_data_dict,
    symbols,
    days=1,
    model_name=LLM_MODEL_NAME,
    temperature=LLM_TEMPERATURE,
):
    """
    Analyzes recent news headlines for each symbol using an LLM via OpenRouter
    to determine sentiment score (1-5).

    1: Very Negative
    2: Rather Negative
    3: Neutral
    4: Rather Positive
    5: Very Positive
    """
    print(f"\n--- News Sentiment Analysis (Last {days} Day(s)) ---")
    sentiment_scores = {}
    if not OPENROUTER_API_KEY:
        print(
            "Error: OPENROUTER_API_KEY environment variable not set. Skipping sentiment analysis."
        )
        return {symbol: "N/A" for symbol in symbols}  # Indicate missing API key

    try:
        # Initialize LLM client based on provider
        if LLM_PROVIDER == "openrouter":
            if not OPENROUTER_API_KEY:
                print(
                    "Error: OPENROUTER_API_KEY environment variable not set. Skipping sentiment analysis."
                )
                return {symbol: "N/A" for symbol in symbols}  # Indicate missing API key

            llm = ChatOpenAI(
                model=model_name,
                openai_api_key=OPENROUTER_API_KEY,
                openai_api_base="https://openrouter.ai/api/v1",
                default_headers={
                    "HTTP-Referer": "https://github.com/nextlevelrobotcom/stock-market-agent ",  # Replace with your actual referer if needed
                    "X-Title": "StockMarketAgent",  # Replace with your app title
                },
                temperature=temperature,
                streaming=False,  # Streaming not needed for single score output
            )
        elif LLM_PROVIDER == "ollama":
            OLLAMA_BASE_URL = os.getenv(
                "OLLAMA_BASE_URL", "http://localhost:11434"
            )  # Uncomment and set if using a non-default base URL
            llm = ChatOllama(
                model=model_name,
                temperature=temperature,
                base_url=OLLAMA_BASE_URL,
            )
        else:
            print(
                f"Error: Unknown LLM_PROVIDER '{LLM_PROVIDER}'. Skipping sentiment analysis."
            )
            return {symbol: "Error" for symbol in symbols}

    except Exception as e:
        print(f"Error initializing LLM client: {e}. Skipping sentiment analysis.")
        return {symbol: "Error" for symbol in symbols}

    system_prompt = """
You are a financial news sentiment analyst. Your task is to analyze a list of news articles for a specific stock symbol and its synonymes that were published within the last day.
The trading strategy is not longterm but always shortterm, so always try to take profits with you and avoid losses whenever possible.
Based *only* on the provided articles, determine the overall sentiment towards the stock symbol. If the stock or its synonymes don't appear in the articles, return a neutral score of 3.
Output *only* a single integer score from 1 to 5, according to the following scale:
1: Very Negative (Strong sell signals, major issues, scandals)
2: Rather Negative (Concerns, missed estimates, analyst downgrades)
3: Neutral (Mixed news, factual reports, minor updates)
4: Rather Positive (Good performance, analyst upgrades, positive developments)
5: Very Positive (Major breakthroughs, strong earnings beat, acquisition news)

If no articles are provided, output 3 (Neutral). Do not provide any explanation or justification, just the single number.
"""

    cutoff_date = None
    # Determine cutoff date using the first available news item's timezone
    for symbol in symbols:
        if symbol in news_data_dict and news_data_dict[symbol]:
            try:
                # Ensure the timestamp is timezone-aware before calculation
                first_news_time = news_data_dict[symbol][-1]["time"]
                if first_news_time.tzinfo is None:
                    # Attempt to localize if naive, assuming UTC or local timezone might be context-dependent
                    # For robustness, consider standardizing timezones during ingestion
                    print(
                        f"Warning: News time for {symbol} is timezone-naive. Assuming UTC."
                    )
                    # Use timezone from datetime module
                    first_news_time = first_news_time.replace(
                        tzinfo=datetime.timezone.utc
                    )

                cutoff_date = first_news_time - timedelta(days=days)
                break  # Found a timezone-aware time, use this cutoff
            except Exception as e:
                print(f"Error processing time for {symbol}: {e}. Trying next symbol.")

    if cutoff_date is None:
        # Fallback if no news items had usable timezones
        print(
            "Warning: Could not determine timezone-aware cutoff date. Using naive UTC comparison."
        )
        cutoff_date = datetime.utcnow() - timedelta(days=days)

    for symbol in symbols:
        print(f"  Analyzing sentiment for {symbol}...")
        recent_headlines = []
        if symbol in news_data_dict and news_data_dict[symbol]:
            for item in news_data_dict[symbol]:
                try:
                    # Ensure item time is comparable to cutoff_date
                    item_time = item["time"]
                    if item_time.tzinfo is None:
                        # Make naive time comparable to potentially timezone-aware cutoff
                        # This comparison might be inaccurate if timezones differ significantly
                        # Use timezone from datetime module
                        item_time_aware = item_time.replace(
                            tzinfo=datetime.timezone.utc
                        )  # Assume UTC if naive
                        cutoff_date_aware = (
                            cutoff_date
                            if cutoff_date.tzinfo
                            else cutoff_date.replace(tzinfo=datetime.timezone.utc)
                        )
                    else:
                        item_time_aware = item_time
                        cutoff_date_aware = cutoff_date  # Already timezone-aware

                    if item_time_aware >= cutoff_date_aware:
                        recent_headlines.append(
                            f"- {item['headline']}: {item['summary']}"
                        )
                except Exception as e:
                    print(f"    Error processing news item time for {symbol}: {e}")

        if not recent_headlines:
            print(f"    No recent news found for {symbol}. Assigning Neutral (3).")
            sentiment_scores[symbol] = 3
            continue

        # Define synonyms for the stock symbol
        synonyms = {
            "AAPL": ["Apple", "iPhone", "MacBook", "iPad", "iOS", "iMac"],
            "MSFT": ["Microsoft", "Windows", "Azure", "Office", "Xbox"],
            "GOOG": ["Google", "Alphabet", "YouTube", "Android"],
            "AMZN": ["Amazon", "AWS", "Kindle", "Prime", "Alexa"],
            "TSLA": ["Tesla", "Elon Musk", "Model S", "Model 3", "Model Y", "Model X"],
            "META": [
                "Meta",
                "Facebook",
                "Instagram",
                "WhatsApp",
                "Oculus",
                "Threads",
                "Metaverse",
            ],
            "NFLX": ["Netflix"],
        }
        # Add synonyms for each symbol to the prompt
        synonyms_text = {}
        for symbol_syn in synonyms:
            synonyms_text[symbol_syn] = ", ".join(synonyms[symbol_syn])

        headlines_text = "\n".join(recent_headlines)
        prompt = f"Stock Symbol: {symbol}\n\nSynonyms: {synonyms_text[symbol]}\n\nRecent Headlines:\n{headlines_text}"

        messages = [SystemMessage(content=system_prompt), HumanMessage(content=prompt)]

        content = "not defined"  # Initialize to avoid reference before assignment
        try:
            response = llm.invoke(messages)
            content = response.content.strip()
            # Try to parse as integer directly
            score = int(content)
            if 1 <= score <= 5:
                sentiment_scores[symbol] = score
                print(f"    Sentiment score for {symbol}: {score}")
            else:
                print(
                    f"    Warning: LLM returned out-of-range score ({score}) for {symbol}. Defaulting to Neutral (3)."
                )
                sentiment_scores[symbol] = 3
        except ValueError as e:
            print(
                f"    ValueError from LLM: {e} - Warning: LLM did not return a valid integer ('{content}') for {symbol}. Defaulting to Neutral (3)."
            )
            sentiment_scores[symbol] = 3
        except Exception as e:
            print(f"    Error invoking LLM for {symbol}: {e}. Assigning Error.")
            sentiment_scores[symbol] = "Error"  # Indicate LLM error

    return sentiment_scores


def send_slack_message(message):
    """Sends a message to a Slack webhook URL."""
    # Set up a WebClient with the Slack OAuth token
    client = WebClient(token=SLACK_TOKEN)

    # Send a message
    client.chat_postMessage(
        channel=SLACK_CHANNEL, text=message, username=SLACK_BOT_NAME
    )


# --- Main Agent Logic ---
def run_agent(
    end_date_str=None,
    use_sentiment=True,
    send_to_slack=True,
    refresh_data=True,
    logger=None,
    symbols=None,
):
    print("--- Starting Stock Analysis Agent ---")
    start_time = datetime.now()

    if symbols is None:
        symbols = SYMBOLS

    # Parse end_date_str if provided, otherwise use current date
    if end_date_str:
        try:
            analysis_end_time = datetime.strptime(end_date_str, "%Y-%m-%d")
            ingest_end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
        except ValueError:
            print("Error: Invalid end_date format. Using current date.")
            analysis_end_time = datetime.now()
            ingest_end_date = datetime.now()
    else:
        analysis_end_time = datetime.now()
        ingest_end_date = datetime.now()

    # 1. Initialize InfluxDB Handler
    influx_handler = InfluxDBHandler()
    if not influx_handler.connect() or not influx_handler.test_connection():
        print("Error: Failed to connect to InfluxDB. Exiting.")
        return

    # 2. Ingest Recent Data
    if refresh_data:
        print("\n--- Data Ingestion ---")
        ingest_start_date = ingest_end_date - timedelta(days=DATA_INGESTION_DAYS)
        ingest_start_str = ingest_start_date.strftime("%Y-%m-%d")
        ingest_end_str = ingest_end_date.strftime("%Y-%m-%d")
        ingestion_successful = True
        for symbol in symbols:
            print(
                f"Ingesting data for {symbol} from {ingest_start_str} to {ingest_end_str}..."
            )
            success = influx_handler.ingest_data(
                symbol, start_date=ingest_start_str, end_date=ingest_end_str
            )
            if not success:
                print(f"Warning: Ingestion failed for {symbol}")
                ingestion_successful = False  # Track if any ingestion failed
        if ingestion_successful:
            print("Data ingestion check completed.")
        else:
            print("Data ingestion check completed with some failures.")

    # 3. Retrieve Data for Analysis
    print("\n--- Data Retrieval for Analysis ---")
    analysis_start_time = analysis_end_time - timedelta(days=ANALYSIS_WINDOW_DAYS)
    analysis_start_time_str = analysis_start_time.isoformat() + "Z"
    analysis_end_time_str_temp = analysis_end_time.isoformat() + "Z"
    analysis_end_time_str = (
        analysis_end_time_str_temp.split("T")[0] + "T23:59:59Z"
    )  # make sure the day is complete

    print(
        f"Retrieving data from {analysis_start_time.date()} to {analysis_end_time.date()}..."
    )
    # IMPORTANT: Current retrieve_data only gets 'close'. Modify stock_data_ingestion.py
    #            to retrieve OHLCV for full technical analysis capabilities.
    only_prices = True
    if use_sentiment:
        only_prices = False
    stock_df, news_data = influx_handler.retrieve_data(
        symbols, analysis_start_time_str, analysis_end_time_str, only_prices=only_prices
    )

    if stock_df.empty:
        print("Warning: No stock data retrieved for analysis. Using mock data.")
        return json.dumps(
            {symbol: 0.0 for symbol in symbols}
        )  # Return default "Hold" recommendation

    print(f"Retrieved stock data shape: {stock_df.shape}")
    if news_data:
        print(f"Retrieved news data for symbols: {list(news_data.keys())}")

    # 4. Perform Technical Analysis
    print("\n--- Technical Analysis ---")
    tech_data_results = calculate_technical_indicators(stock_df, symbols)
    tech_recommendations, tech_scores = get_technical_recommendation(
        tech_data_results, symbols
    )

    # 5. Perform Quantitative Analysis (ARIMA)
    print("\n--- Quantitative Analysis ---")
    quant_recommendations, forecast_ratios = get_quantitative_recommendation(
        stock_df, symbols, ARIMA_ORDER, FORECAST_CHANGE_THRESHOLD
    )

    # 6. Combine Recommendations
    print("\n--- Combining Recommendations ---")
    timeseries_recommendations = combine_recommendations(
        tech_recommendations, quant_recommendations, symbols
    )
    if logger:
        logger.info(f"Combined recommendations: {timeseries_recommendations}")

    # 7. Get News Sentiment Scores
    sentiment_scores = {}
    if use_sentiment:
        if news_data:
            # Use the configured number of days for sentiment analysis
            sentiment_scores = get_news_sentiment_scores(
                news_data, symbols, days=NEWS_SENTIMENT_DAYS
            )
        else:
            print("No news data retrieved, skipping sentiment analysis.")
            sentiment_scores = {symbol: "No News" for symbol in symbols}
    else:
        print("Sentiment analysis disabled.")
        sentiment_scores = {symbol: "Sentiment Analysis Disabled" for symbol in symbols}

    map_sentiment_scores = {1: -1, 2: -0.5, 3: 0, 4: 0.5, 5: 1}

    sentiment_scores_normalized = {
        symbol: map_sentiment_scores.get(score, 0)
        for symbol, score in sentiment_scores.items()
    }

    if logger:
        logger.info(f"Sentiment scores: {sentiment_scores_normalized}")

    # Create final recommendations by combining timeseries_recommendations and sentiment_scores
    final_recommendations = {}
    for symbol in symbols:
        final_recommendations[symbol] = (
            timeseries_recommendations.get(symbol)["Score"]
            + sentiment_scores_normalized[symbol]
        )
    if logger:
        logger.info(f"Final recommendations: {final_recommendations}")

    if send_to_slack:
        # 8. Format Slack Message
        print("\n--- Formatting Slack Message ---")
        message = f"*Stock Analysis Report - {datetime.now():%Y-%m-%d %H:%M}*\n\n"
        message += "*Recommendations & Sentiment:*\n"
        sentiment_map = {
            1: "Very Negative",
            2: "Negative",
            3: "Neutral",
            4: "Positive",
            5: "Very Positive",
            "N/A": "N/A (Check Key)",
            "Error": "LLM Error",
            "No News": "No Recent News",
        }

        for symbol, recs in timeseries_recommendations.items():
            sentiment_score = sentiment_scores.get(
                symbol, "N/A"
            )  # Default if symbol somehow missing
            sentiment_desc = sentiment_map.get(
                sentiment_score, str(sentiment_score)
            )  # Use score directly if not in map
            message += f"- *{symbol}:* {recs['Combined']} (Tech: {recs['Technical']}, Quant: {recs['Quantitative']}, Score: {recs['Score']}, Sentiment: {sentiment_desc})\n"

        # 9. Send Slack Message
        print("\n--- Sending Slack Notification ---")
        send_slack_message(message)
    else:
        print("\n--- Skipping Slack Notification ---")

    end_time = datetime.now()
    print(f"\n--- Agent run finished in {end_time - start_time} ---")

    # 10. Return recommendation scores as JSON, tech scores, and forecast ratios
    return final_recommendations, tech_scores, forecast_ratios


if __name__ == "__main__":
    # Example usage:
    end_date = "2025-04-01"  # Example end date
    use_sentiment = True
    send_to_slack = True
    refresh_data = False
    recommendation_scores_json, tech_scores, forecast_ratios = run_agent(
        end_date_str=end_date,
        refresh_data=refresh_data,
        use_sentiment=use_sentiment,
        send_to_slack=send_to_slack,
    )
    print("\n--- Recommendation Scores ---")
    print(recommendation_scores_json)
    print("\n--- tech_scores ---")
    print(tech_scores)
    print("\n--- forecast_ratios ---")
    print(forecast_ratios)
