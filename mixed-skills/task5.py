import os
import uuid
import datetime
import gradio as gr
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphRecursionError

load_dotenv()

# Tool
@tool
def get_current_date() -> str:
    """Returns today's date in YYYY-MM-DD format. Use when the user asks
    about today's date, the current date, or what day it is."""
    return datetime.date.today().isoformat()

# LLM
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY"),
    reasoning_effort="low",
)

# Agent
memory = MemorySaver()
SYSTEM_PROMPT = """You are a helpful assistant with access to a date tool.
When the user asks about today's date or what day it is, use the get_current_date tool.
Remember previous messages in the conversation."""

agent = create_react_agent(
    model=llm,
    tools=[get_current_date],
    checkpointer=memory,
    prompt=SYSTEM_PROMPT,
)

def chat(message, history, session_id):
    if not message.strip():
        return history, session_id

    config = {
        "configurable": {"thread_id": session_id},
        "recursion_limit": 10,
    }

    try:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": message}]},
            config=config,
        )
        answer = result["messages"][-1].content
    except GraphRecursionError:
        answer = "I couldn't complete this in the allowed steps."
    except Exception as e:
        answer = f"An error occurred: {e}"

    history = history + [{"role": "user", "content": message}, {"role": "assistant", "content": answer}]
    return history, session_id

with gr.Blocks(title="Task 5 — Chatbot with Memory and Date Tool") as demo:
    session_id = gr.State(value=lambda: str(uuid.uuid4()))
    gr.Markdown("# Chatbot with Memory and Date Tool")

    chatbot = gr.Chatbot(height=400)
    msg_box = gr.Textbox(placeholder="Ask anything...", label="Your message")
    submit_btn = gr.Button("Send")

    submit_btn.click(
        chat,
        inputs=[msg_box, chatbot, session_id],
        outputs=[chatbot, session_id]
    ).then(lambda: "", outputs=msg_box)

    msg_box.submit(
        chat,
        inputs=[msg_box, chatbot, session_id],
        outputs=[chatbot, session_id]
    ).then(lambda: "", outputs=msg_box)

demo.launch()