import logging
from qdrant_client import QdrantClient
from phi.model.groq import Groq
from phi.agent import Agent
from phi.embedder.ollama import OllamaEmbedder  # Using OllamaEmbedder
from phi.knowledge.pdf import PDFUrlKnowledgeBase
from phi.vectordb.qdrant import QdrantVectorDb  # Correct import for QdrantVectorDb

# Initialize logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set URL and connect to the Qdrant client
url = "http://172.20.10.64:6333"
client = QdrantClient(url=url)

# Function to list all collections and their vector counts
def list_collections():
    try:
        # Fetch all collections from Qdrant
        response = client.get_collections()

        # Access collections from the response
        collections = response.collections

        if not collections:
            logger.info("No collections found.")
        else:
            for collection in collections:
                collection_name = collection.name  # Access collection name
                logger.info(f"Collection: {collection_name}")

                # Fetch and display the vector count of each collection
                summary = client.get_collection(collection_name)

                logger.info(f" - Vectors count: {summary.vectors_count}")

    except Exception as e:
        logger.error(f"Error fetching collections: {e}")

# Now, move to build for the RAG Agent
vector_db = QdrantVectorDb(
    client=client,  # Use the same Qdrant client
    collection_name="IT_infrastructure",  # Specify the correct collection name
    embedder=OllamaEmbedder(model="openhermes"),  # Using OllamaEmbedder for embeddings
)

# Create a knowledge base from a PDF using OllamaEmbedder for embeddings
knowledge_base = PDFUrlKnowledgeBase(
    urls=["http://example.com/sample.pdf"],  # Replace with a valid PDF URL list
    vector_db=vector_db,  # Pass the QdrantVectorDb
)

# Comment out after first run as the knowledge base is loaded
knowledge_base.load(upsert=True)  # Ensure the PDF URL is accessible and valid

# Define the agent
agent = Agent(
    model=Groq(id="groq-model-1"),  # Use a proper Groq model ID if this is a real model
    knowledge=knowledge_base,  # Add the knowledge base to the agent
    show_tool_calls=True,
    markdown=True,
)

# Interact with the agent
agent.print_response("About Motion detectors based on infra-red technology", stream=True)
