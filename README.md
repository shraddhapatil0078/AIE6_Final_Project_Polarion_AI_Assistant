# Polarion AI Assistant

A specialized AI assistant for Polarion ALM configuration, customization, and administration. This project uses advanced RAG (Retrieval-Augmented Generation) techniques to provide accurate and context-aware responses about Polarion documentation.

## Features

- 🤖 AI-powered responses to Polarion-related queries
- 📚 Context-aware answers based on Polarion documentation
- 🔍 Advanced document processing with OCR capabilities
- 📊 Response evaluation system for quality assurance
- 💬 Interactive chat interface using Chainlit

## Project Structure

```
.
├── app.py                 # Main application with Chainlit integration
├── eval.py               # Evaluation system for response quality
├── polarion_doc_indexer/ # Document processing and indexing module
│   └── indexer.py        # PDF processing and vector store indexing
├── pyproject.toml        # Project dependencies and configuration
└── chainlit.md          # Chainlit configuration
```

## Prerequisites

- Python 3.10 or higher
- OpenAI API key
- Qdrant API key and cluster URL
- Tesseract OCR (for document processing)

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd polarion-ai-assistant
```

2. Install dependencies:
```bash
pip install -e .
```

3. Set up environment variables:
Create a `.env` file with the following variables:
```
OPENAI_API_KEY=your_openai_api_key
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_CLUSTER_URL=your_qdrant_cluster_url
```

## Usage

1. Start the application:
```bash
chainlit run app.py
```

2. Open your browser and navigate to the provided local URL (typically http://localhost:8000)

3. Start chatting with the Polarion AI Assistant!

## Document Processing

The system includes a document processing pipeline that:
- Processes PDF documents with OCR capabilities
- Extracts text and images
- Creates embeddings for semantic search
- Stores processed documents in Qdrant vector database

To process new documents:
```bash
python polarion_doc_indexer/indexer.py
```

## Evaluation System

The assistant includes an evaluation system that:
- Assesses response faithfulness
- Measures relevance to user queries
- Provides quality scores for responses

## Dependencies

Key dependencies include:
- chainlit: For the chat interface
- langchain: For RAG implementation
- qdrant-client: For vector storage
- pymupdf: For PDF processing
- pytesseract: For OCR capabilities
- langchain-openai: For OpenAI integration

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Add your license information here]
