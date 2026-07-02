import os
import uuid
import base64
import shutil
import gradio as gr
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from agent import run_agent_with_trace

load_dotenv()

embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
)

def index_pdf(pdf_file):
    if pdf_file is None:
        return "No file uploaded."

    from tools_rag import vectorstore
    try:
        vectorstore.delete_collection()
    except:
        pass

    filename = os.path.basename(pdf_file)
    loader = PyPDFLoader(pdf_file)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(pages)

    for chunk in chunks:
        chunk.metadata["source"] = filename
        chunk.metadata["page"] = chunk.metadata.get("page", 0) + 1

    Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory="./chroma_store",
        collection_name="hybridsight",
    )
    return f"Indexed {filename} — {len(chunks)} chunks stored."

def image_to_data_uri(filepath: str) -> str:
    from PIL import Image
    import io
    
    img = Image.open(filepath)
    img = img.convert("RGB")
    img.thumbnail((512, 512))  # resize to max 512x512
    
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=50)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"

def chat(message, image, history, session_id):
    if not message.strip() and image is None:
        return history, session_id, ""

    full_message = message

    # Handle image separately — get description first, then pass text to agent
    if image is not None:
        try:
            data_uri = image_to_data_uri(image)
            print("DATA URI LENGTH:", len(data_uri))
            from groq import Groq
            vision_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
            vision_response = vision_client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"The user asks: {message}\n\nDescribe this image in detail."},
                        {"type": "image_url", "image_url": {"url": data_uri}},
                    ],
                }],
            )
            image_description = vision_response.choices[0].message.content
            print("IMAGE DESCRIPTION:", image_description)
            full_message = f"{message}\n\n[Image description: {image_description}]"
            print("FULL MESSAGE SENT TO AGENT:", full_message[:200])
        except Exception as e:
            print("VISION ERROR:", e)
            full_message = f"{message}\n\n[Image could not be processed: {e}]"

    answer, trace = run_agent_with_trace(full_message, session_id)

    if answer.startswith("An error occurred"):
        session_id = str(uuid.uuid4())

    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": answer}
    ]
    return history, session_id, trace

with gr.Blocks(title="HybridSight") as demo:
    session_id = gr.State(value=lambda: str(uuid.uuid4()))
    gr.Markdown("# 👁️ HybridSight — RAG + Web + Vision Agent")

    with gr.Row():
        pdf_upload = gr.File(label="Upload a PDF", file_types=[".pdf"])
        image_upload = gr.Image(label="Upload an image", type="filepath")

    index_status = gr.Textbox(label="Indexing status", interactive=False)
    pdf_upload.change(
        fn=lambda f: index_pdf(f.name) if f else "No file uploaded.",
        inputs=pdf_upload,
        outputs=index_status
    )

    chatbot = gr.Chatbot(height=420, label="Conversation")
    msg_box = gr.Textbox(placeholder="Ask anything...", label="Your message")
    submit_btn = gr.Button("Send", variant="primary")

    with gr.Accordion("🔍 Agent Reasoning Trace", open=False):
        trace_box = gr.Textbox(
            label="Tools called during last response",
            lines=6,
            interactive=False
        )

    submit_btn.click(
        chat,
        inputs=[msg_box, image_upload, chatbot, session_id],
        outputs=[chatbot, session_id, trace_box]
    ).then(lambda: "", outputs=msg_box)

    msg_box.submit(
        chat,
        inputs=[msg_box, image_upload, chatbot, session_id],
        outputs=[chatbot, session_id, trace_box]
    ).then(lambda: "", outputs=msg_box)

if __name__ == "__main__":
    demo.launch()