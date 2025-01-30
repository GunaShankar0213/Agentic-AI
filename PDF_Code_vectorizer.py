import pickle
import logging
from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Define file paths
pdf_options = {
   # "BPS": 'C:/Users/Guna Shankar/Downloads/Org/rag-demo/release/BPS.pdf',
    #"Portescap": 'C:/Users/Guna Shankar/Downloads/Org/rag-demo/release/Porterscap.pdf',
    "IT Infrastructure": 'C:/Users/Guna Shankar/Downloads/Org/rag-demo/release/IT Infrastructure.pdf',
    "SCM": 'C:/Users/Guna Shankar/Downloads/Org/rag-demo/release/SCM.pdf',
    "BXS": 'C:/Users/Guna Shankar/Downloads/Org/rag-demo/release/BxS_Area_Location_Constraint.pdf',
}

vectorstore_paths = {
    #"BPS": 'C:/Users/Guna Shankar/Downloads/Temp/Vectorizers/all_doc_vectorstore.pkl',
    #"Portescap": 'C:/Users/Guna Shankar/Downloads/Temp/Vectorizers/all_doc_porter_vectorstore.pkl',
    "IT Infrastructure": 'C:/Users/Guna Shankar/Downloads/Temp/Vectorizers/it_infra.pkl',
    "SCM": 'C:/Users/Guna Shankar/Downloads/Temp/Vectorizers/scm.pkl',
    "BXS": 'C:/Users/Guna Shankar/Downloads/Temp/Vectorizers/bxs.pkl'
}

# Load Ollama Embedding Model
embedder = OllamaEmbeddings(model='bge-m3')

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file using PyPDFLoader."""
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    return documents  # Returns list of Document objects

def process_and_store_embeddings():
    """Process PDFs, generate embeddings, and store vectors locally."""
    for doc_name, pdf_path in pdf_options.items():
        logging.info(f"Processing: {doc_name}")

        # Extract text from the PDF
        documents = extract_text_from_pdf(pdf_path)

        # Split text into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_documents(documents)  # Splitting Documents, not plain text

        # Generate embeddings with progress bar
        logging.info(f"Generating embeddings for {doc_name}... ({len(chunks)} chunks)")
        embeddings = [embedder.embed_query(chunk.page_content) for chunk in tqdm(chunks)]

        # Store embeddings in a pickle file
        vector_path = vectorstore_paths[doc_name]
        with open(vector_path, "wb") as f:
            pickle.dump(embeddings, f)

        logging.info(f"Stored embeddings at: {vector_path}")
        logging.info(f"Embeddings Shape for {doc_name}: {len(embeddings)} vectors generated\n")

if __name__ == "__main__":
    process_and_store_embeddings()
