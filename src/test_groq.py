import os

from dotenv import load_dotenv
from groq import Groq


# Load .env
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError(
        "GROQ_API_KEY not found. Check your .env file."
    )


# Create Groq client
client = Groq(
    api_key=api_key
)


# Test request
response = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[
        {
            "role": "system",
            "content": (
                "You are a helpful customer support assistant."
            )
        },
        {
            "role": "user",
            "content": (
                "Say hello in one short sentence."
            )
        }
    ],
    temperature=0.2,
    max_completion_tokens=100
)


print(
    response.choices[0].message.content
)