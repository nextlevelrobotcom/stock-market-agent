from influxdb_client import InfluxDBClient
from datetime import datetime
from dotenv import load_dotenv
import requests
import os

load_dotenv()

token = os.getenv("TOKEN")
org = os.getenv("ORG")
bucket = os.getenv("BUCKET")

# Setup
client = InfluxDBClient(url="http://localhost:8086", token=token, org=org)
delete_api = client.delete_api()

# Time range: delete everything from the beginning to far future
start = "2020-01-01T00:00:00Z"
stop = "2021-06-01T00:00:00Z"

symbols_delete = [
    "MSFT",
    "GOOG",
    "AMZN",
    "TSLA",
    "META",
    "NFLX",
]  # Example symbols

for symbol in symbols_delete:
    # Predicate to match the symbol
    predicate = f'_measurement="market_news" AND symbol="{symbol}"'

    # Perform delete
    delete_api.delete(start, stop, predicate, bucket=bucket, org=org)
    print(f"Deleted data for {symbol} from {start} to {stop}")

print("Data deletion completed.")
