import os
from langchain_community.llms import GPT4All
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# --- Configuration ---
INDEX_DIR = "faiss_index"
MODEL_PATH = r"C:\Users\SANMATI\.cache\gpt4all\orca-mini-3b-gguf2-q4_0.gguf"

# --- Embeddings (local) ---
emb = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
print("✅ Using local Hugging Face embeddings")

# --- Load FAISS vectorstore ---
if not os.path.exists(INDEX_DIR):
    raise FileNotFoundError(f"❌ FAISS index not found at {INDEX_DIR}/")
vectorstore = FAISS.load_local(INDEX_DIR, emb, allow_dangerous_deserialization=True)

# --- Retriever (restrict context size) ---
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 2}  # ✅ fetch fewer docs
)

# --- Memory (short history only) ---
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    k=1  # ✅ only last user + agent turn
)

# --- LLM (GPT4All local model) ---
llm = GPT4All(
    model=MODEL_PATH,
    backend="gptj",
    allow_download=False,
    verbose=False,
    device="cpu"
)
print("🧠 Using GPT4All local model (Orca Mini 3B, CPU mode)")

# --- Conversational Retrieval Chain ---
chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=False
)

print("\n🤖 Local AI Agent is ready! Type your question or 'exit' to quit.\n")

# --- Truncate helper ---
def truncate_text(text: str, max_words: int = 500) -> str:
    """Ensure input text fits inside context window."""
    words = text.split()
    return " ".join(words[-max_words:])

# --- Chat loop ---
while True:
    query = input("You: ").strip()
    if query.lower() in ("exit", "quit"):
        print("👋 Goodbye!")
        break

    try:
        short_query = truncate_text(query)
        result = chain.invoke({"question": short_query})
        print("\nAgent:", result["answer"])
    except Exception as e:
        print("⚠️ Error:", e)
