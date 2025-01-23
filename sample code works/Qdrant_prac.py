import os
import logging
from tqdm import tqdm  
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from phi.model.groq import Groq
from dotenv import load_dotenv
load_dotenv() # to call API

# Loading the sample data

# # Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# # Specify the folder path containing the PDFs
# folder_path = "/home/guna_shankar/Agentic_Ai/sample code works"

# # Set your desired Qdrant collection name
# url = "http://172.20.10.64:6333"
# collection_name = "my_collection_pdf"  # Updated collection name

# logger.info("Starting the process")

# # Load the embedding model
# try:
#     embeddings = OllamaEmbeddings(model='bge-m3')
#     logger.info("Embedding model successfully loaded")
# except Exception as e:
#     logger.error(f"Error loading embedding model: {e}")
#     raise

# # Get all PDF files from the folder
# pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]

# # Initialize counters
# success_count = 0
# failure_count = 0

# # Process each PDF file
# for filename in tqdm(pdf_files, desc="Processing PDFs", unit="file"):
#     pdf_file = os.path.join(folder_path, filename)  # Get the full path of the PDF file
#     logger.info(f"Processing PDF: {pdf_file}")

#     try:
#         # Load the PDF document
#         loader = PyPDFLoader(pdf_file)
#         documents = loader.load()
#         logger.info(f"Loaded document: {pdf_file}")

#         # Split the document into chunks
#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
#         texts = text_splitter.split_documents(documents)
#         logger.info(f"Split the document into {len(texts)} chunks")

#         # Insert the embeddings into Qdrant
#         if texts:
#             qdrant = Qdrant.from_documents(
#                 texts,  
#                 embeddings,
#                 url=url,
#                 prefer_grpc=False,
#                 collection_name=collection_name
#             )
#             logger.info(f"Vector DB updated with {len(texts)} embeddings from document: {filename}")
#             success_count += 1 
#         else:
#             logger.warning(f"No texts found for document: {filename}")

#     except Exception as e:
#         # Log errors and continue
#         logger.error(f"Error processing {pdf_file}: {e}")
#         failure_count += 1  
#         continue

# # Final summary
# logger.info(f"Processing complete: {success_count} files processed successfully, {failure_count} files failed.")

# To View the Documents

from qdrant_client import QdrantClient

collection_name = "my_collection_pdf"  # The collection you want to query
client = QdrantClient(url="http://172.20.10.64:6333", port=6333)

# Retrieve all points (embeddings and metadata) from the collection
def retrieve_all_data(collection_name):
    # Query the collection for all points
    response = client.scroll(
        collection_name=collection_name,
        limit=10,  # Adjust as needed; this defines the number of points per scroll
    )
    
    # Print the structure of the response to debug
    print("Response Structure:", response)
    
    all_points = []
    while response:
        # If the response is a tuple, we need to adjust accordingly
        if isinstance(response, tuple):
            # Usually the first item in the tuple is the data, we can check if it contains "result"
            data = response[0]
        else:
            # Else, we assume it's a dictionary
            data = response
        
        # Extract points from the response (which should be a list of points)
        if "result" in data:
            all_points.extend(data["result"])
        
        # Move to the next batch (scrolling)
        if isinstance(data, dict) and "next_page_offset" in data:
            response = client.scroll(
                collection_name=collection_name,
                offset=data["next_page_offset"],
                limit=100,
            )
        else:
            break
    
    return all_points

# Display retrieved data
def display_data(points):
    for point in points:
        # Extract relevant information from the point's payload
        page_number = point["payload"].get("page_number", "Unknown")
        text = point["payload"].get("text", "No text")
        
        print(f"Page: {page_number}")
        print(f"Text: {text[:300]}...")  # Print the first 300 characters of text for preview
        print("-" * 50)

# Main process to retrieve and display data
def retrieve_and_display(collection_name):
    print(f"Retrieving data from Qdrant collection: {collection_name}")
    
    points = retrieve_all_data(collection_name)
    
    if points:
        print(f"Retrieved {len(points)} points from the collection.")
        display_data(points)
    else:
        pass

# Call the function to retrieve and display data from the collection
#retrieve_and_display(collection_name)


#################  Lets initialize with Agent ########################

# import os
# from phi.knowledge.pdf import PDFUrlKnowledgeBase
# from phi.vectordb.qdrant import Qdrant
# import os
# import typer
# from typing import Optional
# from rich.prompt import Prompt

# from phi.agent import Agent
# from phi.knowledge.pdf import PDFUrlKnowledgeBase
# from phi.vectordb.qdrant import Qdrant

# # No need for an API key or authentication when using a local Qdrant server
# qdrant_url = "http://172.20.10.64:6333"
# collection_name = "my_collection_pdf"

# # Create the Qdrant vector database object without using API key
# vector_db = Qdrant(
#     collection=collection_name,
#     url=qdrant_url,
#     # Local Qdrant URL without the API key
# )
# pdf_path = "/home/guna_shankar/Agentic_Ai/sample code works/"
# # Define the knowledge base with the local server's PDF file path
# knowledge_base = PDFUrlKnowledgeBase(
#     path=pdf_path,
#     vector_db=vector_db,
# )

# # Load the knowledge base with or without upserting based on your need
# knowledge_base.load(recreate=True, upsert=True)

# os.environ["GROQ_API_KEY"] = "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" 

# agent = Agent(
#     model=Groq(id="llama-3.3-70b-versatile",api_key=os.environ["GROQ_API_KEY"]),
#     # Add the knowledge base to the agent
#     knowledge=knowledge_base,
#     show_tool_calls=True,
#     markdown=True,
# )

import os
import ollama
from phi.knowledge.pdf import PDFKnowledgeBase
from phi.vectordb.qdrant import Qdrant
from phi.document import Document
from phi.agent import Agent

# Define paths and URLs
pdf_path = "/home/guna_shankar/Agentic_Ai/sample code works/"
qdrant_url = "http://172.20.10.64:6333"
collection_name = "my_collection_pdf"

# Initialize the knowledge base for local PDFs and Qdrant client
knowledge_base = PDFKnowledgeBase(path=pdf_path, vector_db=None)
vector_db = Qdrant(collection=collection_name, url=qdrant_url)

# Function to generate embeddings using Ollama model
def generate_embeddings(text: str):
    try:
        # Request embeddings from Ollama model
        response = ollama.chat(model="llama3.2:1b", messages=[{"role": "user", "content": text}])
        embedding = response.get("embedding")
        if embedding is None:
            print("No embedding found in response.")
            return None
        return embedding
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return None

# Function to store embeddings in Qdrant
def store_embeddings_in_qdrant(doc_id: str, embedding):
    try:
        # Store the embedding in Qdrant using the document ID
        vector_db.upsert(vectors=[embedding], ids=[doc_id])
        print(f"Embedding for {doc_id} stored successfully.")
    except Exception as e:
        print(f"Error storing embedding in Qdrant: {e}")

# Extract text from PDFs, generate embeddings, and store them in Qdrant
def process_pdfs_and_store_embeddings():
    documents = []
    for doc_list in knowledge_base.document_lists:
        for doc in doc_list:
            # Print the document object to inspect its structure
            print(f"Document structure: {dir(doc)}")
            
            # Try to access the correct text attribute based on the document structure
            if hasattr(doc, "content"):  # If 'content' exists
                text = doc.content
            elif hasattr(doc, "text"):  # If 'text' exists
                text = doc.text
            else:
                print(f"Cannot find text in document {doc.id}")
                continue

            # Generate embeddings for the extracted text
            embedding = generate_embeddings(text)

            # If embedding is successfully generated, store it in Qdrant
            if embedding:
                store_embeddings_in_qdrant(doc.id, embedding)

    print("Embedding process completed.")

# Initialize agent (this part depends on your use case, this will create an agent object)
def initialize_agent():
    os.environ["GROQ_API_KEY"] = "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"
    agent = Agent(
        model=Groq(id="llama-3.3-70b-versatile", api_key=os.environ["GROQ_API_KEY"]),
        knowledge=knowledge_base,
        show_tool_calls=True,
        markdown=True,
    )
    return agent

# Main function to handle the entire process
def main():
    # Load the knowledge base (optional to upsert data)
    knowledge_base.load(recreate=True, upsert=True)

    # Process PDFs, generate embeddings, and store them in Qdrant
    process_pdfs_and_store_embeddings()

    # Initialize the agent
    agent = initialize_agent()

    # Interact with the agent (Example)
    user_input = input("Ask the agent something: ")
    agent.print_response(user_input, stream=True)

if __name__ == "__main__":
    main()
