import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

def ingest_documents(pdf_path: str):
    """
    Load a PDF, split it into chunks, embed, and save into Chroma vector DB.
    Run this function once for each new PDF you want to include in RAG.
    """
    print(f"Loading PDF: {pdf_path}")
    loader = PyPDFLoader(os.path.join("data", pdf_path))
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    db = Chroma.from_documents(chunks, embeddings, persist_directory="chroma_db")
    db.persist()

    print("Ingestion complete! Data saved to 'chroma_db'.")

def load_retriever():
    """
    Load embeddings and Chroma DB (cached version).
    If the DB already exists, it will reuse it instead of rebuilding.
    """
    if not os.path.exists("chroma_db"):
        os.makedirs("chroma_db")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
    print("✅ Database loaded successfully.")
    return db.as_retriever(search_kwargs={"k": 3})


retriever = load_retriever()

print("🚀 Initializing Groq LLM (llama-3.1-8b-instant)...")
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=groq_api_key
)

chat_history = []

def format_history(history):
    return "\n".join([f"{h['role'].capitalize()}: {h['content']}" for h in history])

# Prompt template
prompt = ChatPromptTemplate.from_template(
    """
    You are RAGBot — a smart, friendly, and conversational assistant that answers
    questions based on the provided document context and the ongoing chat history.

    You should:
    - Maintain natural, human-like flow in conversation.
    - Use the chat history to recall details (like the user's name or previous topics).
    - Be concise — don't restate obvious things or overexplain.
    - Answer only from the provided document context when relevant.
    - If a question is unrelated to the documents, politely say so — but still respond
      helpfully or conversationally.
    - Avoid repeating your name or previous messages unless necessary.
    - Keep your tone warm, confident, and approachable.
    - Whenever the user asks some keyword related to the document, try to respond using the context.
    - Don't keep calling out the user's name, just use it naturally if it fits.

    Chat history:
    {chat_history}

    context:
    {context}

    question:
    {question}
    """
)

# Build RAG pipeline
rag_chain = (
    {
        "context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)),
        "question": RunnablePassthrough(),
        "chat_history": lambda _: format_history(chat_history)
    }
    | prompt
    | llm
)

print("\n🟢 Ask your questions below.")
print("Type 'exit' to stop.\n")

while True:
    query = input("🟡 You: ")
    if query.lower() in ["exit", "quit", "bye"]:
        print("👋 Goodbye!")
        break
    
    response = rag_chain.invoke(query)
    print("💬 RAGBot:", response.content, "\n")

    chat_history.append({"role": "user", "content": query})
    chat_history.append({"role": "assistant", "content": response.content})
