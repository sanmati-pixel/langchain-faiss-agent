import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# ---- Load environment variables ----
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

DATA_DIR = Path("data")
INDEX_DIR = Path("faiss_index")
INDEX_DIR.mkdir(exist_ok=True)

# ---- Step 1: Read all .txt files ----
docs = []
for file in DATA_DIR.glob("*.txt"):
    text = file.read_text(encoding="utf-8")
    docs.append(Document(page_content=text, metadata={"source": file.name}))

# ---- Step 2: Split documents into chunks ----
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
texts, metadatas = [], []

for d in docs:
    parts = splitter.split_text(d.page_content)
    for p in parts:
        texts.append(p)
        metadatas.append(d.metadata)

print(f"📄 Loaded {len(docs)} documents, created {len(texts)} chunks.")

# ---- Step 3: Create embeddings ----
# ---- Step 3: Create free local embeddings ----
from langchain_community.embeddings import HuggingFaceEmbeddings
emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
print("⚙️ Using free Hugging Face Embeddings (offline, no API key needed)")

# ---- Step 4: Create FAISS index ----
vectorstore = FAISS.from_texts(texts, emb, metadatas=metadatas)

# ---- Step 5: Save the index ----
vectorstore.save_local(str(INDEX_DIR))
print(f"💾 Saved FAISS index with {len(texts)} chunks to {INDEX_DIR}/")
