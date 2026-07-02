import os
import datetime
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphRecursionError

from tools_rag import search_documents
from tools_vision import describe_image

load_dotenv()

# MODEL
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
)

# TOOLS
@tool
def search_web(query: str) -> str:
    """Search the web for real-time information. Use for current news,
    recent events, live data, or anything published after 2024.
    Do NOT use for background knowledge, history, or definitions."""
    searcher = DuckDuckGoSearchRun()
    return searcher.run(query)

@tool
def search_wikipedia(query: str) -> str:
    """Look up encyclopaedic information on Wikipedia. Use for historical
    facts, scientific concepts, notable people, and background context.
    Do NOT use for current events or real-time information."""
    wiki = WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=1000)
    return wiki.run(query)

tools = [search_web, search_documents]

# SYSTEM PROMPT
TODAY = datetime.date.today().isoformat()
SYSTEM_PROMPT = f"""You are HybridSight, a research assistant with two tools.
Today's date is {TODAY}.

ROUTING RULES:
1. If the user asks about uploaded documents, 'my notes', or 'the file', use search_documents.
2. For everything else — current events, general knowledge — use search_web.
3. If an image description is provided in the message, use it directly to answer the question.
4. Always state which tool provided each piece of information.
5. If no tool returns useful information, say so honestly.
"""

# AGENT
memory = MemorySaver()
agent = create_react_agent(
    model=llm,
    tools=tools,
    checkpointer=memory,
    prompt=SYSTEM_PROMPT,
)

# INFERENCE WITH TRACE
def run_agent_with_trace(user_input: str, session_id: str) -> tuple:
    trace_log = []
    final_answer = ""
    config = {
        "configurable": {"thread_id": session_id},
        "recursion_limit": 12,
    }

    try:
        for event in agent.stream(
            {"messages": [{"role": "user", "content": user_input}]},
            config=config,
            stream_mode="values",
        ):
            last = event["messages"][-1]
            if hasattr(last, "tool_calls") and last.tool_calls:
                for tc in last.tool_calls:
                    trace_log.append(f"-> Tool: {tc['name']}\n   Input: {tc['args']}")
            elif last.type == "ai" and not last.tool_calls:
                final_answer = last.content

    except GraphRecursionError:
        final_answer = "I couldn't finish within the step limit. Try rephrasing your question."
    except Exception as e:
        final_answer = f"An error occurred: {e}"

    trace_str = "\n\n".join(trace_log) if trace_log else "No tools were called."
    return final_answer, trace_str
