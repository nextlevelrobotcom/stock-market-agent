## def main
# test_slack.py

import os
from slack_sdk import WebClient

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def main():
    message = "This is my first Slack message from Python!"

    # Set up a WebClient with the Slack OAuth token
    client = WebClient(token=os.getenv("SLACK_TOKEN"))

    # Send a message
    client.chat_postMessage(
        channel="investment", text=message, username="Investment consultant"
    )


if __name__ == "__main__":
    # Run the script
    main()
