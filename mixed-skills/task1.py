from groq import Groq
import os  #A built-in Python module that lets you interact with your computer's operating system, allowing your code to read environment variables
from dotenv import load_dotenv  #A function from the python-dotenv library that reads a .env file and loads its hidden key-value pairs into your system's environment

load_dotenv()
client = Groq(api_key = os.getenv("GROQ_API_KEY"))

system_prompt = input("System: ")
user_prompt = input("User: ")

response = client.chat.completions.create(
    model = "llama-3.3-70b-versatile",
    messages= [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
)

print("\nAssistant", response.choices[0].message.content)