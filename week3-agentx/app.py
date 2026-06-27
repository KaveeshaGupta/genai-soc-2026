import uuid
import gradio as gr
from agent import run_agent_with_trace

def chat(message, history, session_id):
    if not message.strip():
        return history, session_id, ""
    
    import uuid
    answer, trace = run_agent_with_trace(message, session_id)
    
    # if error, reset session to clear broken state
    if answer.startswith("An error occurred"):
        session_id = str(uuid.uuid4())
    
    history = history + [{"role": "user", "content": message}, {"role": "assistant", "content": answer}]
    return history, session_id, trace

with gr.Blocks(title="AgentX — Research Agent") as demo:
    session_id = gr.State(value=lambda: str(uuid.uuid4()))

    gr.Markdown("# 🤖 AgentX\nA research agent with web search, Wikipedia, and memory.")

    chatbot = gr.Chatbot(height=420, label="Conversation")
    msg_box = gr.Textbox(placeholder="Ask anything...", label="Your question")
    submit_btn = gr.Button("Send", variant="primary")

    with gr.Accordion("🔍 Agent Reasoning Trace", open=False):
        trace_box = gr.Textbox(
            label="Tools called during last response",
            lines=6,
            interactive=False
        )

    submit_btn.click(
        chat,
        inputs=[msg_box, chatbot, session_id],
        outputs=[chatbot, session_id, trace_box]
    ).then(lambda: "", outputs=msg_box)

    msg_box.submit(
        chat,
        inputs=[msg_box, chatbot, session_id],
        outputs=[chatbot, session_id, trace_box]
    ).then(lambda: "", outputs=msg_box)

if __name__ == "__main__":
    demo.launch()