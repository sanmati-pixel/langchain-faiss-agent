import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- Configuration ---
TXT_DIR = "data"              # Folder where your .txt files are stored
INDEX_DIR = "faiss_index"     # Output folder for FAISS index
CHUNK_SIZE = 400              # Small chunks to fit within model token limit
CHUNK_OVERLAP = 50            # Slight overlap to maintain context continuity

# --- Load Embeddings ---
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
print("✅ Using local Hugging Face embeddings")

# --- Collect all text files ---
txt_files = [f for f in os.listdir(TXT_DIR) if f.lower().endswith(".txt")]
if not txt_files:
    raise FileNotFoundError(f"❌ No .txt files found in {TXT_DIR}/ folder!")

# --- Load all text documents ---
docs = []
for txt_file in txt_files:
    path = os.path.join(TXT_DIR, txt_file)
    print(f"📄 Loading: {txt_file}")
    loader = TextLoader(path, encoding="utf-8")
    docs.extend(loader.load())

# --- Split into smaller chunks ---
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)
texts = splitter.split_documents(docs)
print(f"✂️ Split into {len(texts)} small chunks (avg {CHUNK_SIZE} tokens each)")

# --- Build FAISS vectorstore ---
if not os.path.exists(INDEX_DIR):
    os.makedirs(INDEX_DIR)

vectorstore = FAISS.from_documents(texts, embeddings)
vectorstore.save_local(INDEX_DIR)

print(f"\n✅ New FAISS index built successfully at '{INDEX_DIR}/'")
print("Now you can run 'run_agent.py' again — the 2048-token error should be gone.")
