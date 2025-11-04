import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Folder containing your files
DOCS_FOLDER = "data"
INDEX_DIR = "faiss_index"

# Load embedding model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

docs = []

# Load documents from the data folder
for file in os.listdir(DOCS_FOLDER):
    path = os.path.join(DOCS_FOLDER, file)

    if file.endswith(".pdf"):
        print(f"📄 Loading PDF: {file}")
        loader = PyPDFLoader(path)
        docs.extend(loader.load())

    elif file.endswith(".txt"):
        print(f"📝 Loading Text: {file}")
        loader = TextLoader(path, encoding="utf-8")
        docs.extend(loader.load())

print(f"✅ Total documents loaded: {len(docs)}")

if not docs:
    raise ValueError("❌ No documents found to index. Please check your data folder.")

# Create FAISS vector store
print("⚙️ Building FAISS vector store...")
vectorstore = FAISS.from_documents(docs, embeddings)

# Save the index
vectorstore.save_local(INDEX_DIR)
print(f"✅ FAISS index successfully saved in '{INDEX_DIR}' folder!")
