# Medical_AiBot 🩺

Medical_AiBot is an experimental conversational AI assistant designed to help answer medical questions by combining a LangChain-enabled Retrieval-Augmented Generation (RAG) backend with a lightweight React frontend.

## 🚀 Features
- Conversational RAG powered by LangChain and a vector store (Pinecone)
- PDF document ingestion and chunking for knowledge retrieval
- Simple React chat UI with Markdown rendering
- Environment-driven configuration (.env) for safe API key management

---

## 📦 Tech Stack
- Backend: Python, Flask, LangChain, Pinecone
- Embeddings: HuggingFace / sentence-transformers
- Frontend: React (Create React App)

---

## 🔧 Quick Start
Follow these steps to get the project running locally.

### Prerequisites
- Python 3.10+ (or a recent 3.x version)
- Node.js 18+ and npm
- Pinecone account and API key
- OpenAI API key (or compatible LLM provider configured in env)

### Backend
1. Create and activate a virtual environment:
	```bash
	python -m venv .venv
	# Windows
	.venv\Scripts\activate
	# macOS / Linux
	source .venv/bin/activate
	```
2. Install backend dependencies:
	```bash
	pip install -r Backend/requirements.txt
	```
3. Create a `.env` in the `Backend/` directory with the following variables:
	```env
	OPENAI_API_KEY=your_openai_api_key_here
	PINECONE_API_KEY=your_pinecone_api_key_here
	```
4. Index your PDF documents (uploads are read from `Backend/data/`):
	```bash
	python Backend/store_index.py
	```
	This will create / populate a Pinecone index named `medical-assistant`.
5. Run the Flask server:
	```bash
	python Backend/app.py
	```
	The backend listens on http://localhost:5000 by default and exposes a POST `/ask` endpoint that expects JSON `{ "question": "..." }`.

### Frontend
1. Install dependencies and run:
	```bash
	cd medibot_frontend
	npm install
	npm start
	```
2. Open http://localhost:3000 in your browser and interact with the chat UI.

> The React frontend sends requests to `http://localhost:5000/ask` by default. Change this in `src/components/Chatbot.js` if your API is hosted elsewhere.

---

## 🗂️ Project Structure (short)
- Backend/
  - `app.py` — Flask API and RAG chain
  - `store_index.py` — loads PDFs, creates embeddings, and stores vectors in Pinecone
  - `src/` — helper modules for loading, splitting, and embedding documents
- medibot_frontend/ — React UI

---

## ⚠️ Security & Notes
- **Do not** commit `.env` or any API keys to source control.
- Pinecone and OpenAI usage may incur costs; monitor usage and quotas.

---

## Contributing
Contributions are welcome — open issues or PRs for bug fixes, improvements, or new features. Please include clear descriptions and, when applicable, reproducible steps.

## License
This repository does not include a license file. Add a LICENSE (e.g., MIT) if you plan to open source the project.

---

If you'd like, I can add badges, runtime examples, screenshots, or CI setup next. ✅