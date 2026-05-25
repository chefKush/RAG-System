# test_groq.py
# Purpose: Confirm we can reach Groq's API and get a response from an LLM.
# If this works, our foundation is solid and we can start building RAG.

import os                          # Built-in: lets us read environment variables
from dotenv import load_dotenv     # Reads the .env file we created
from groq import Groq  # type: ignore         # Groq's official Python SDK
 
# Step 1: Load the .env file into the environment.
# After this line runs, os.getenv() can see GROQ_API_KEY.
load_dotenv()

# Step 2: Pull the API key out of the environment.
# os.getenv returns None if the variable isn't set, so we check for that.
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    # Fail loudly and clearly if the key is missing.
    # A cryptic error 50 lines later is much worse than this clear message.
    raise ValueError(
        "GROQ_API_KEY not found. "
        "Check that .env exists in this folder and contains GROQ_API_KEY=..."
    )

print("API key loaded successfully (first 10 chars):", api_key[:10] + "...")

# Step 3: Create a Groq client.
# Think of this as opening a phone line to Groq's servers.
# We pass our key so Groq knows who's calling.
client = Groq(api_key=api_key)

# Step 4: Send a chat completion request.
# This is the actual API call — the moment your laptop talks to Groq.
print("\nSending request to Groq...")

response = client.chat.completions.create(
    # The model name — Llama 3.1 8B is fast, free, and good enough for learning.
    model="llama-3.1-8b-instant",

    # Messages are a list of turns in a conversation.
    # "system" sets the assistant's behavior; "user" is what we're asking.
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant. Keep answers short."
        },
        {
            "role": "user",
            "content": "Explain what a vector embedding is, in 2 sentences."
        }
    ],

    # Controls randomness: 0 = deterministic, 1 = creative.
    # Low values are better for factual RAG answers later.
    temperature=0.7,
)

# Step 5: Extract the actual text reply from the response object.
# The response is a structured object; the text lives at this path.
answer = response.choices[0].message.content

print("\n=== Groq's Response ===")
print(answer)
print("=======================")

# Step 6: Print some metadata so you see what actually happened under the hood.
# This is the "visibility" mindset we'll keep using throughout the project.
print(f"\nModel used:      {response.model}")
print(f"Prompt tokens:   {response.usage.prompt_tokens}")
print(f"Response tokens: {response.usage.completion_tokens}")
print(f"Total tokens:    {response.usage.total_tokens}")