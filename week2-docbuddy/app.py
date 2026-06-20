import gradio as gr
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from groq import Groq
import os
import shutil
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
)

vectorstore = None

def index_documents(pdf_files):
    global vectorstore

    if os.path.exists("./chroma_store"):
        shutil.rmtree("./chroma_store")

    all_chunks = []
    for pdf_file in pdf_files:
        filename = os.path.basename(pdf_file.name)
        loader = PyPDFLoader(pdf_file.name)
        pages = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_documents(pages)
        for chunk in chunks:
            chunk.metadata["source"] = filename
            chunk.metadata["page"] = chunk.metadata.get("page", 0) + 1
        all_chunks.extend(chunks)
        print(f"Loaded {filename}: {len(chunks)} chunks")

    vectorstore = Chroma.from_documents(
        documents=all_chunks,
        embedding=embedding_model,
        persist_directory="./chroma_store",
        collection_name="docbuddy",
    )

    total = vectorstore._collection.count()
    print(f"Indexed {total} total chunks")
    return f"✅ Indexed {len(pdf_files)} document(s) — {total} total chunks"

def ask(question, history):
    global vectorstore

    if vectorstore is None:
        if os.path.exists("./chroma_store"):
            vectorstore = Chroma(
                persist_directory="./chroma_store",
                embedding_function=embedding_model,
                collection_name="docbuddy",
            )
        else:
            history = history or []
            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": "⚠️ No documents indexed yet. Please upload PDFs first."})
            return "", history, ""

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    chunks = retriever.invoke(question)

    context = "\n\n".join([
        f"[Source {i}: {doc.metadata.get('source','?')}, page {doc.metadata.get('page','?')}]\n{doc.page_content}"
        for i, doc in enumerate(chunks, 1)
    ])

    messages = [
        {"role": "system", "content": (
            "Answer using ONLY the context below. "
            "If the answer isn't in the context, say: 'I don't have that information in the uploaded documents.' "
            "After your answer, add a 'Sources:' line citing the source filename and page number."
        )},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0,
    )
    answer = response.choices[0].message.content

    context_display = "\n\n".join([
        f"**[{doc.metadata.get('source','?')} · Page {doc.metadata.get('page','?')}]**\n{doc.page_content[:300]}..."
        for doc in chunks
    ])

    history = history or []
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": answer})
    return "", history, context_display

with gr.Blocks(title="DocBuddy Pro") as demo:
    gr.Markdown("# DocBuddy Pro — Q&A Over Multiple PDFs")

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=420)
            msg_box = gr.Textbox(placeholder="Ask a question about your documents...")
            ask_btn = gr.Button("Ask")

        with gr.Column(scale=2):
            file_upload = gr.File(
                file_count="multiple",
                file_types=[".pdf"],
                label="Upload PDFs"
            )
            index_btn = gr.Button("Index Documents")
            status_label = gr.Markdown("No documents indexed yet.")

            with gr.Accordion("🔍 Retrieved Context (last query)", open=False):
                context_display = gr.Markdown()

    index_btn.click(
        fn=index_documents,
        inputs=file_upload,
        outputs=status_label
    )

    ask_btn.click(
        fn=ask,
        inputs=[msg_box, chatbot],
        outputs=[msg_box, chatbot, context_display]
    )

demo.launch()