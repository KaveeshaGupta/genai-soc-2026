import gradio as gr
from groq import Groq
import os
import json
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

PERSONAS = {
    "Technical Explainer": {
        "system_prompt": "You are a clear, jargon-free technical explainer. You break down complex topics for beginners. Keep responses under 150 words unless asked to expand.",
        "few_shot_examples": [
            {"role": "user", "content": "What is an API?"},
            {"role": "assistant", "content": "An API is a messenger between two apps. When you log into a website using Google, that website sends a request to Google's API asking 'is this person who they say they are?' Google replies yes or no. The website never sees your Google password — the API handles the conversation between them."},
        ],
        "output_format": "text"
    },
    "Debate Coach": {
        "system_prompt": "You are a debate coach. For any topic, argue BOTH sides clearly. Label them 'FOR:' and 'AGAINST:'. Give 2 strong points per side. Be concise.",
        "few_shot_examples": [
            {"role": "user", "content": "Should everyone learn to code?"},
            {"role": "assistant", "content": "FOR:\n1. Coding teaches logical thinking applicable to any field.\n2. Tech literacy is increasingly essential in every industry.\n\nAGAINST:\n1. Not everyone's career benefits from coding.\n2. Low-code tools are making manual coding less necessary."},
        ],
        "output_format": "text"
    },
    "Code Reviewer": {
        "system_prompt": 'You are an expert code reviewer. Respond ONLY with valid JSON matching this schema exactly. No preamble, no markdown fences: {"issues": ["..."], "suggestions": ["..."], "severity": "low|medium|high"}',
        "few_shot_examples": [
            {"role": "user", "content": "def add(a, b):\n    return a - b"},
            {"role": "assistant", "content": '{"issues": ["Function named add but subtracts instead"], "suggestions": ["Change return a - b to return a + b"], "severity": "high"}'},
        ],
        "output_format": "json"
    },
    "Creative Writer": {
        "system_prompt": "You are a vivid, imaginative creative writer. Use rich sensory details, metaphors, and flowing prose. Every response should feel like a page from a novel.",
        "few_shot_examples": [
            {"role": "user", "content": "Describe a rainy day."},
            {"role": "assistant", "content": "The rain arrived like a whispered secret, tapping at windowpanes with patient, insistent fingers. The street below dissolved into a mirror of silver and grey, each puddle capturing a broken reflection of amber streetlights above."},
        ],
        "output_format": "text"
    }
}

def build_messages(persona_name, api_history, user_message):
    persona = PERSONAS[persona_name]
    messages = [{"role": "system", "content": persona["system_prompt"]}]
    for example in persona["few_shot_examples"]:
        messages.append(example)
    for msg in api_history:
        messages.append(msg)
    messages.append({"role": "user", "content": user_message})
    return messages

def respond(message, history, persona_name, temperature):
    history = history or []
    
    api_history = []
    for entry in history:
        api_history.append({"role": "user", "content": entry["content"] if isinstance(entry, dict) else entry[0]})
        api_history.append({"role": "assistant", "content": entry["content"] if isinstance(entry, dict) else entry[1]})
    
    messages = build_messages(persona_name, api_history, message)

    stream = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=temperature,
        stream=True,
    )

    accumulated = ""
    history = history + [{"role": "user", "content": message}, {"role": "assistant", "content": ""}]
    
    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        accumulated += delta

        persona = PERSONAS[persona_name]
        if persona["output_format"] == "json":
            try:
                parsed = json.loads(accumulated)
                rendered = f"**Severity:** {parsed.get('severity', 'N/A').upper()}\n\n**Issues:**\n" + \
                           "\n".join(f"- {i}" for i in parsed.get('issues', [])) + \
                           "\n\n**Suggestions:**\n" + \
                           "\n".join(f"- {s}" for s in parsed.get('suggestions', []))
                history[-1]["content"] = rendered
            except json.JSONDecodeError:
                history[-1]["content"] = accumulated
        else:
            history[-1]["content"] = accumulated

        yield "", history

with gr.Blocks(title="PromptForge") as demo:
    gr.Markdown("# PromptForge — Multi-Mode AI Assistant")

    with gr.Row():
        persona_dropdown = gr.Dropdown(
            choices=list(PERSONAS.keys()),
            value="Technical Explainer",
            label="Select Mode"
        )
        temperature_slider = gr.Slider(
            minimum=0.0,
            maximum=1.5,
            value=0.7,
            step=0.1,
            label="Temperature"
        )

    with gr.Accordion("Active System Prompt", open=False):
        system_prompt_display = gr.Markdown(
            value=PERSONAS["Technical Explainer"]["system_prompt"]
        )

    chatbot = gr.Chatbot()
    user_input = gr.Textbox(placeholder="Type your message here...", label="Your Message")
    send_btn = gr.Button("Send")

    def update_system_prompt(persona_name):
        return PERSONAS[persona_name]["system_prompt"]

    persona_dropdown.change(
        fn=update_system_prompt,
        inputs=persona_dropdown,
        outputs=system_prompt_display
    )

    send_btn.click(
        fn=respond,
        inputs=[user_input, chatbot, persona_dropdown, temperature_slider],
        outputs=[user_input, chatbot]
    )

demo.launch()