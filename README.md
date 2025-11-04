# langchain-faiss-agent
🧠 AI Agent using RAG (Retrieval-Augmented Generation)

This project demonstrates how to build a local AI Agent capable of reading .txt documents and answering questions using LangChain, FAISS, and Hugging Face embeddings.

🚀 Features

✅ Loads .txt documents from a local folder
✅ Splits data into small chunks for accurate retrieval
✅ Embeds data using all-MiniLM-L6-v2
✅ Stores and retrieves context using FAISS vector DB
✅ Generates responses with OpenAI / HuggingFace models
✅ Fully local & version-compatible pipeline

🧩 Architecture
User Query
   ↓
FAISS Vector Search ←→ Embeddings (HuggingFace)
   ↓
Context Sent to LLM (OpenAI or HF)
   ↓
Final Response


🛠️ Setup Instructions
1. Clone Repository
git clone https://github.com/<your-username>/AI-Agent.git
cd AI-Agent

2. Create Virtual Environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate (Windows)

3. Install Requirements
pip install -r requirements.txt

4. Add Your Data

Place your .txt files inside the /data folder.

5. Build the Vector Index
python build_index_small_chunks.py

6. Run the AI Agent
python agent.py

🧠 Skills Demonstrated

LangChain 🧩

FAISS Vector DB 💾

Hugging Face Embeddings 🤗

OpenAI / LLM Integration 🔮

Debugging & Version Management ⚙️

Local AI Infrastructure Design 💡

⚙️ Tech Stack
Component	Library
Language	Python
LLM	OpenAI GPT / HuggingFace
Embeddings	all-MiniLM-L6-v2
Vector Store	FAISS
Framework	LangChain
🧰 Folder Structure
AI-Agent/
├── data/ (your text files)
├── index/ (FAISS database)
├── build_index_small_chunks.py
├── agent.py
├── requirements.txt
├── README.md
└── .gitignore

📸 Screenshots (Optional)

You can add a screenshot of:

Terminal showing successful index creation

Your AI agent giving an answer
(using Markdown: ![alt text](screenshot.png))

🏆 Author

👩‍💻 Sanmati Pol
📍 Data Science & AI Enthusiast
💬 “Building real-world AI tools that actually work!”


