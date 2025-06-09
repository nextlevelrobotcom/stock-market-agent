import json
import logging
import csv
import os
from datetime import datetime, timedelta
import pytz
from stock_agent import run_agent
from stock_data_ingestion import InfluxDBHandler
import random

# --- Simulation Parameters ---
END_DATE = "2025-02-07"
INITIAL_BUDGET = 10000.0
RANDOM_TRADING_LIMIT = 0.6
TRADING_COST = 2.0  # Cost per trade (buy or sell)
SYMBOLS = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA", "META", "NFLX"]
MAX_TRADES_PER_DAY = len(SYMBOLS)  # Maximum trades per day
PORTFOLIO = {}  # Initialize empty portfolio
BUDGET = INITIAL_BUDGET
total_transaction_costs = 0.0
influx_handler = None
current_stock_price = {}

# --- Logging Configuration ---
now = datetime.now()
dt_string = now.strftime("%Y_%m_%d_%H_%M")
trading_dir = "trading"
os.makedirs(trading_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename=f"{trading_dir}/trading_simulation_{dt_string}.log",
)
logger = logging.getLogger(__name__)


def trading_simulation(
    start_date_str,
    end_date_str,
    random_simulation=False,
    use_sentiment=False,
    initial_portfolio=None,
    initial_budget=None,
    initial_transaction_costs=None,
):
    global BUDGET, PORTFOLIO, influx_handler, total_transaction_costs
    if initial_portfolio is not None:
        PORTFOLIO = initial_portfolio
    if initial_budget is not None:
        BUDGET = initial_budget
    if initial_transaction_costs is not None:
        total_transaction_costs = initial_transaction_costs
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    start_date = pytz.utc.localize(start_date)  # Make it UTC aware
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    end_date = pytz.utc.localize(end_date)  # Make it UTC aware
    current_date = start_date
    while current_date <= end_date:
        current_date_str = current_date.strftime("%Y-%m-%d")
        logger.info(f"--- {current_date_str} ---")

        influx_handler = InfluxDBHandler()
        if not influx_handler.connect():
            logger.error("Failed to connect to InfluxDB. Using mock stock prices.")
            return

        # Get current stock prices
        if current_date.weekday() == 5:  # Saturday
            logger.info(
                "Saturday: Skipping trading for today. Date = " + current_date_str
            )
            current_date += timedelta(days=1)
            continue
        elif current_date.weekday() == 6:  # Sunday
            logger.info(
                "Sunday: Skipping trading for today. Date = " + current_date_str
            )
            current_date += timedelta(days=1)
            continue
        stock_data, _ = influx_handler.retrieve_data(
            SYMBOLS,
            current_date_str + "T00:00:00Z",
            current_date_str + "T23:59:59Z",
            only_prices=True,
        )
        continue_trade = True
        for symbol in SYMBOLS:
            if (
                stock_data.empty
                or symbol not in stock_data.columns
                or current_date not in stock_data.index
            ):
                logger.warning(
                    f"Could not retrieve stock price for {symbol} on {current_date.strftime('%Y-%m-%d')}. Using mock price."
                )
                continue_trade = False
                break
            else:
                current_stock_price[symbol] = stock_data[symbol].loc[current_date]

        if not continue_trade:
            logger.info(
                "Skipping trading for today due to missing stock data. Date = "
                + current_date_str
            )
            current_date += timedelta(days=1)
            continue

        if random_simulation:
            recommendation_scores = {
                symbol: random.uniform(-0.75, 0.75) for symbol in SYMBOLS
            }
        else:
            recommendation_scores = {}

        # 2. Trading Logic
        trades_today = 0

        # First sort portfolio by stock quantity (highest first)
        sorted_portfolio = sorted(PORTFOLIO.items(), key=lambda x: x[1], reverse=True)

        score = None
        # Process sell recommendations first
        for symbol, quantity in sorted_portfolio:
            if trades_today >= MAX_TRADES_PER_DAY:
                logger.info(f"Reached maximum trades for today ({MAX_TRADES_PER_DAY}).")
                break

            if random_simulation:
                sell_threshold = -RANDOM_TRADING_LIMIT
            else:
                if use_sentiment:
                    sell_threshold = -0.5
                else:
                    sell_threshold = -0.5

            if not random_simulation:
                if quantity > 0:
                    # 1. Get Recommendations from Stock Agent
                    recommendation_scores_symbol, _, _ = run_agent(
                        end_date_str=current_date_str,
                        use_sentiment=use_sentiment,
                        send_to_slack=False,
                        refresh_data=False,
                        logger=logger,
                        symbols=[symbol],
                    )
                    recommendation_scores[symbol] = recommendation_scores_symbol[symbol]
                else:
                    continue
                score = recommendation_scores_symbol.get(symbol, 0)
            else:
                score = recommendation_scores.get(symbol, 0)

            if score <= sell_threshold and quantity > 0:
                try:
                    stock_price = current_stock_price[symbol]
                    BUDGET += quantity * stock_price - TRADING_COST
                    total_transaction_costs += TRADING_COST
                    trades_today += 1
                    logger.info(
                        f"Sold {quantity} shares of {symbol} at {stock_price:.2f}. New budget: {BUDGET:.2f}"
                    )
                    del PORTFOLIO[symbol]
                except Exception as e:
                    logger.error(f"Error selling {symbol}: {e}")

        # get the lowest cost stock price from the portfolio
        lowest_cost_stock_price = min(
            [
                current_stock_price[symbol]
                for symbol in SYMBOLS
                if current_stock_price[symbol] > 0
            ],
            default=0,
        )
        highest_cost_stock_price = max(
            [
                current_stock_price[symbol]
                for symbol in SYMBOLS
                if current_stock_price[symbol] > 0
            ],
            default=10000,
        )
        # print to logger
        logger.info(
            f"Lowest cost stock price in portfolio: {lowest_cost_stock_price:.2f}; Budget: {BUDGET:.2f}"
        )
        # print to logger
        logger.info(
            f"Highest cost stock price in portfolio: {highest_cost_stock_price:.2f}; Budget: {BUDGET:.2f}"
        )

        if random_simulation:
            buy_threshold = RANDOM_TRADING_LIMIT
        else:
            if use_sentiment:
                buy_threshold = 1.0
            else:
                buy_threshold = 0.5

        # Process buy recommendations only if we have significant budget left
        if BUDGET > TRADING_COST and BUDGET > highest_cost_stock_price:
            if not random_simulation:
                ## get highest forecast ratio
                recommendation_scores_symbol, tech_scores, forecast_ratios = run_agent(
                    end_date_str=current_date_str,
                    use_sentiment=False,
                    send_to_slack=False,
                    refresh_data=False,
                    logger=logger,
                    symbols=SYMBOLS,
                )

            else:
                ## random simulation: randomly generate forecast ratios
                forecast_ratios = {
                    symbol: random.uniform(0.9, 1.1) for symbol in SYMBOLS
                }

            symbols_ordered_by_forecast = sorted(
                forecast_ratios.items(), key=lambda item: item[1], reverse=True
            )

            for symbol, forecast in symbols_ordered_by_forecast:
                ## only consider symbols that have a forecast ratio > 1.0
                if forecast < 1.0:
                    ## here we break, since we ordered the symbols by forecast ratio
                    ## and we only want to buy stocks that have a forecast ratio > 1.0
                    break

                # if recommendation score already exists, skip other wise get the scores
                if recommendation_scores.get(symbol) is None:
                    # 1. Get Recommendations from Stock Agent
                    recommendation_scores_symbol, _, _ = run_agent(
                        end_date_str=current_date_str,
                        use_sentiment=use_sentiment,
                        send_to_slack=False,
                        refresh_data=False,
                        logger=logger,
                        symbols=[symbol],
                    )
                    recommendation_scores[symbol] = recommendation_scores_symbol[symbol]

                    ## break the loop if there is one buy signal
                    if recommendation_scores[symbol] >= buy_threshold:
                        break
                else:
                    ## here we break because we already have the recommendation score of the largest forecast ratio
                    break

            for symbol, score in recommendation_scores.items():
                if trades_today >= MAX_TRADES_PER_DAY:
                    break

                if score >= buy_threshold:
                    try:
                        stock_price = current_stock_price[symbol]
                        if stock_price > 0:
                            max_quantity = int((BUDGET - TRADING_COST) / stock_price)
                            if max_quantity > 0:
                                if symbol not in PORTFOLIO:
                                    PORTFOLIO[symbol] = 0
                                PORTFOLIO[symbol] += max_quantity
                                BUDGET -= max_quantity * stock_price + TRADING_COST
                                total_transaction_costs += TRADING_COST
                                trades_today += 1
                                logger.info(
                                    f"Bought {max_quantity} shares of {symbol} at {stock_price:.2f}. New budget: {BUDGET:.2f}"
                                )
                    except Exception as e:
                        logger.error(f"Error buying {symbol}: {e}")

        # 4. Print Final Results
        logger.info("\n--- Final Portfolio ---")
        logger.info(PORTFOLIO)
        # Calculate portfolio value
        portfolio_value = 0
        for symbol, quantity in PORTFOLIO.items():
            portfolio_value += quantity * current_stock_price[symbol]
        logger.info(f"--- Final Portfolio Value: {portfolio_value:.2f} ---")
        logger.info(f"--- Final Budget: {BUDGET:.2f} ---")
        total_value = portfolio_value + BUDGET
        logger.info(f"--- Total Value: {total_value:.2f} ---")

        # Save trading summary to CSV
        with open(
            f"{trading_dir}/trading_summary_{dt_string}.csv", mode="a", newline=""
        ) as csv_file:
            csv_writer = csv.writer(csv_file)
            if current_date == start_date:
                header = (
                    ["Date"]
                    + SYMBOLS
                    + [
                        "Portfolio Value",
                        "Final Budget",
                        "Total Value",
                        "Total Transaction Costs",
                    ]
                )
                csv_writer.writerow(header)
            portfolio_data = [PORTFOLIO.get(symbol, 0) for symbol in SYMBOLS]
            csv_writer.writerow(
                [current_date_str]
                + portfolio_data
                + [portfolio_value, BUDGET, total_value, total_transaction_costs]
            )

        current_date += timedelta(days=1)


if __name__ == "__main__":
    use_sentiment = True
    random_simulation = False

    # 2021-09-15,0,0,338,0,0,0,0,48848.26171875,30.431255340576172,48878.692974090576,310.0
    # trading_simulation(
    #    "2021-10-07",
    #    "2021-12-10",
    #    # "2022-02-01",
    #    # initial_portfolio={"AAPL": 1, "MSFT": 180, "AMZN": 3},
    #    initial_portfolio={"GOOG": 338},
    #    initial_budget=30.431255340576172,
    #    initial_transaction_costs=310.0,
    #    random_simulation=random_simulation,
    #    use_sentiment=use_sentiment,
    # )

    # 2021-11-05,0,0,0,0,177,0,0,72103.31158447266,26.361492156982422,72129.67307662964,314.0
    # trading_simulation(
    #    "2021-10-07",
    #    "2021-12-10",
    #    # "2022-02-01",
    #    # initial_portfolio={"AAPL": 1, "MSFT": 180, "AMZN": 3},
    #    initial_portfolio={"GOOG": 338},
    #    initial_budget=30.431255340576172,
    #    initial_transaction_costs=310.0,
    #    random_simulation=random_simulation,
    #    use_sentiment=use_sentiment,
    # )

    # 2021-12-10,0,0,0,1,1,0,95,58618.91946411133,9.555416107177734,58628.474880218506,322.0
    # trading_simulation(
    #    "2021-12-10",
    #    "2022-02-07",
    #    # "2022-02-01",
    #    # initial_portfolio={"AAPL": 1, "MSFT": 180, "AMZN": 3},
    #    initial_portfolio={"NFLX": 95, "TSLA": 1, "AMZN": 1},
    #    initial_budget=9.555416107177734,
    #    initial_transaction_costs=322.0,
    #    random_simulation=random_simulation,
    #    use_sentiment=use_sentiment,
    # )

    # Date,AAPL,MSFT,GOOG,AMZN,TSLA,META,NFLX,Portfolio Value,Final Budget,Total Value,Total Transaction Costs
    # 2021-11-05,0,0,0,0,177,0,0,72103.31158447266,26.361492156982422,72129.67307662964,314.0
    # trading_simulation(
    #    "2021-10-07",
    #    "2021-12-10",
    #    # "2022-02-01",
    #    # initial_portfolio={"AAPL": 1, "MSFT": 180, "AMZN": 3},
    #   initial_portfolio={"GOOG": 338},
    #    initial_budget=30.431255340576172,
    #   initial_transaction_costs=310.0,
    #    random_simulation=random_simulation,
    #    use_sentiment=use_sentiment,
    # )

    ## critical point where the large drops of tesla happens
    # 2021-11-05,0,0,0,0,177,0,0,72103.31158447266,26.361492156982422,72129.67307662964,314.0
    # trading_simulation(
    #    "2021-11-05",
    #    "2021-11-09",
    #    # "2022-02-01",
    #    # initial_portfolio={"AAPL": 1, "MSFT": 180, "AMZN": 3},
    #    initial_portfolio={"TSLA": 177},
    #    initial_budget=26.361492156982422,
    #    initial_transaction_costs=314.0,
    #    random_simulation=random_simulation,
    #    use_sentiment=use_sentiment,
    # )

    trading_simulation(
       "2020-04-01",
       "2025-04-01",
       random_simulation=random_simulation,
       use_sentiment=use_sentiment,
    )

    ## continue the simulation from the last date
    # 2023-03-17,0,0,0,0,185,0,0,33324.05090332031,83.41104507446289,33407.461948394775,310.0
    #trading_simulation(
    #    "2023-03-17",
    #    "2025-04-01",
    #    initial_portfolio={"TSLA": 185},
    #    initial_budget=83.41104507446289,
    #    initial_transaction_costs=310.0,
    #    random_simulation=random_simulation,
    #    use_sentiment=use_sentiment,
    #)

    ## new test
    # 2021-12-22,0,0,0,0,0,0,115,70637.59887695312,164.7081069946289,70802.30698394775,274.0
    # trading_simulation(
    #    "2021-12-22",
    #    "2022-02-09",
    #    initial_portfolio={"NFLX": 115},
    #    initial_budget=164.7081069946289,
    #    initial_transaction_costs=274.0,
    #    random_simulation=random_simulation,
    #    use_sentiment=use_sentiment,
    # )
