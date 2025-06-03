import os
import fitz  # PyMuPDF
import uuid
import base64
import openai
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
from PIL import Image
import pytesseract
import io
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# Load environment variables
load_dotenv()
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_CLUSTER_URL = os.getenv("QDRANT_CLUSTER_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Configure OpenAI
openai.api_key = OPENAI_API_KEY
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
client = OpenAI(api_key=OPENAI_API_KEY)

def image_to_markdown(image_data):
    base64_image = base64.b64encode(image_data).decode('utf-8')
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Read and extract the full visible content from this image without skipping or summarizing anything. "
                            "Preserve the exact structure using Markdown format including tables, lists, headers, paragraphs, and indentation. "
                            "Do not omit or ignore any lines, even if they appear at the bottom or sides. Include every word and line exactly as seen. "
                            "Only return the Markdown content."
                        )
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        max_tokens=4096
    )
    return response.choices[0].message.content

def extract_text_with_ocr(image_data):
    image = Image.open(io.BytesIO(image_data))
    return pytesseract.image_to_string(image)

def process_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    markdown_docs = []
    
    for i, page in enumerate(doc):
        pix = page.get_pixmap(matrix=fitz.Matrix(2.5, 2.5))
        image_data = pix.tobytes("png")
        
        markdown_content = image_to_markdown(image_data)
        ocr_text = extract_text_with_ocr(image_data)
        
        last_ocr_line = ocr_text.strip().splitlines()[-1] if ocr_text.strip() else ""
        if last_ocr_line and last_ocr_line not in markdown_content:
            markdown_content += f"\n\n<!-- OCR Fallback -->\n{last_ocr_line}"
            
        heading = ""
        for line in markdown_content.split('\n'):
            if line.strip().startswith('#'):
                heading = line.strip().lstrip('#').strip()
                break
                
        markdown_docs.append({
            "content": markdown_content,
            "ocr_text": ocr_text,
            "metadata": {
                "source": "Polarion Admin Guide",
                "page_number": i + 1,
                "page_width": page.rect.width,
                "page_height": page.rect.height,
                "heading": heading
            }
        })
    
    return markdown_docs

def create_embeddings(markdown_docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    docs_to_embed = []
    
    for doc in markdown_docs:
        chunks = splitter.create_documents(
            texts=[doc["content"]],
            metadatas=[doc["metadata"]]
        )
        docs_to_embed.extend(chunks)
    
    return docs_to_embed

def setup_qdrant_collection(collection_name):
    qdrant_client = QdrantClient(url=QDRANT_CLUSTER_URL, api_key=QDRANT_API_KEY)
    
    if collection_name not in [col.name for col in qdrant_client.get_collections().collections]:
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE)
        )
    
    return qdrant_client

def index_documents(docs_to_embed, collection_name):
    qdrant_client = setup_qdrant_collection(collection_name)
    points = []
    
    for chunk in docs_to_embed:
        vector = embedding_model.embed_query(chunk.page_content)
        payload = chunk.metadata.copy()
        payload["page_content"] = chunk.page_content
        points.append(models.PointStruct(
            id=str(uuid.uuid4()),
            payload=payload,
            vector=vector
        ))
    
    qdrant_client.upsert(collection_name=collection_name, points=points)

def main():
    pdf_path = "./Cropped_Admin_guide_1.pdf"
    collection_name = "polarion_admin_guide_chunks_1"
    
    print("Starting PDF processing...")
    
    # Process PDF and create markdown docs
    markdown_docs = process_pdf(pdf_path)
    print("PDF processing completed. Number of pages processed:", len(markdown_docs))
    
    # Create embeddings
    print("Creating embeddings...")
    docs_to_embed = create_embeddings(markdown_docs)
    print("Embeddings created. Number of documents to embed:", len(docs_to_embed))
    
    # Index documents in Qdrant
    print("Indexing documents in Qdrant...")
    index_documents(docs_to_embed, collection_name)
    print("Indexing completed.")

if __name__ == "__main__":
    main()