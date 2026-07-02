from langchain_core.tools import tool
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
)

vectorstore = Chroma(
    persist_directory="./chroma_store",
    embedding_function=embedding_model,
    collection_name="hybridsight",
)

@tool
def search_documents(query: str) -> str:
    """Search the user's uploaded documents for information relevant to
    the query. Use this when the user asks about content from a PDF they
    uploaded, or references 'the document', 'my notes', or 'the file'.
    Do NOT use this for general knowledge or current events."""
    if vectorstore._collection.count() == 0:
        return "No documents uploaded yet. Please upload a PDF first."

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    chunks = retriever.invoke(query)

    if not chunks:
        return "No relevant content found in the uploaded documents."

    return "\n\n".join([
        f"[p.{c.metadata.get('page','?')}] {c.page_content}"
        for c in chunks
    ])