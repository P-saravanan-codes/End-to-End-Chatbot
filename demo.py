import os
from dotenv import load_dotenv

# Load the variables from the .env file
load_dotenv()

# Retrieve the API key
api_key = os.getenv("GEMINI_API_KEY")

# Check if the environment variable exists
if not api_key:
    raise ValueError("Error: MY_API_KEY is missing from the environment or .env file.")
else:
    print("API Key loaded successfully!")
    # Proceed with your API call using api_key
