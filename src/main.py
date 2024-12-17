#!/usr/bin/env python3
# Import necessary modules for logging, asynchronous programming, and environment variable management.
import logging
logger = logging.getLogger(__name__)

import asyncio
import os
from dotenv import load_dotenv
from pathlib import Path

# Get the folder this file is in:
this_file_folder = os.path.dirname(os.path.realpath(__file__))
# Get the parent folder of this file's folder:
parent_folder = os.path.dirname(this_file_folder)

# Load environment variables from the specified .env file in the bot_envs directory.
load_dotenv(Path(parent_folder) / "bot_envs/.env_onboarding")

# Import the Slack bot instance from the slackbot module.
from slackbot import slack_bot

# Define the main asynchronous function to start the Slack bot.
async def main():
    await slack_bot.start()

# Check if the script is being run as the main module and execute the main function.
if __name__ == "__main__":
    asyncio.run(main())

import logging
logger = logging.getLogger(__name__)

import asyncio
import os
from dotenv import load_dotenv
from pathlib import Path

# Get the folder this file is in:
this_file_folder = os.path.dirname(os.path.realpath(__file__))
# Get the parent folder of this file's folder:
parent_folder = os.path.dirname(this_file_folder)

load_dotenv(Path(parent_folder) / "bot_envs/.env_onboarding")

from slackbot import slack_bot

async def main():
    await slack_bot.start()

if __name__ == "__main__":
    asyncio.run(main())
