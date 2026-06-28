import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent

load_dotenv()

# Define the tool
@tool
def add_numbers(a: int, b: int) -> int:
    """Adds two integers and returns the result."""
    return a + b

# Set up the LLM
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY"),
    reasoning_effort="low",
)

# Create the agent
agent = create_react_agent(
    model=llm,
    tools=[add_numbers],
)

# Invoke the agent
result = agent.invoke({"messages": [("user", "What is 12 + 15?")]})
print(result["messages"][-1].content)