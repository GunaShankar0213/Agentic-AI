# import logging
# import os
# import pickle
# from langchain.embeddings import OllamaEmbeddings
# import ollama  # Ensure Ollama is installed for integration

# # Path to the folder containing the vector files
# vectorstore_folder = 'C:/Users/Guna Shankar/Downloads/Temp/Vectorizers'

# # Keywords to trigger vector search
# TESTING_KEYWORDS = {"testing", "unit test", "unit processing", "test case"}

# # Prompt templates (same as previous)
# PROMPTS = {
#     "detailed": """
#     You are an expert in software testing for a scheduling application designed to handle various task management operations, such as task assignment, prioritization, and resource management. Your task is to generate comprehensive and well-documented test cases for the scheduling application's functionality, considering different test scenarios based on the provided feature.

#     In your response, provide test cases that cover common functionalities like task assignment, task transitions (e.g., "On Hold"), resource allocation, and system visibility. Include all relevant sections such as test case IDs, scenarios, steps, expected results, and any pre-conditions.

#     Each test case should cover the following aspects:
#     1. **Test Case Scenario** – A short description of the feature being tested.
#     2. **Test Case ID** – A unique identifier for each test case.
#     3. **Test Case** – The name of the test case.
#     4. **Test Case Description** – A detailed explanation of what is being tested.
#     5. **Pre-condition** – Any requirements or setup needed before executing the test.
#     6. **Test Steps** – The detailed steps to execute the test.
#     7. **Input Data** – Example input data used for the test.
#     8. **Expected Results** – The anticipated outcome of the test.

#     Ensure that you offer examples of input data that will test the application in realistic and edge case scenarios. For example, test scenarios can involve normal task status changes or edge cases like assigning tasks to unavailable resources, handling conflicts between tasks, or scheduling tasks in overlapping time frames.

#     **Context:**
#     {context}

#     **Question:**
#     {question}

#     **Guidance:**
#     Provide highly detailed test cases to address different scenarios in the scheduling application, ensuring coverage of core functionalities. Each test case should be easy to follow, properly formatted, and aligned with the best practices in scheduling applications.
#     """,
    
#     "concise": """
#     You are an expert in software testing for a scheduling application, and your role is to generate focused and clear test cases that cover core functionalities, such as task transitions, resource management, and task assignment. 

#     Provide the test case structure in a brief format with:
#     1. **Test Case Scenario** – A brief description of the test being tested.
#     2. **Test Case ID** – A unique identifier.
#     3. **Test Case** – The name of the test case.
#     4. **Test Case Description** – A short explanation of the functionality being tested.
#     5. **Pre-condition** – Any setup required before running the test.
#     6. **Test Steps** – A concise list of actions to execute the test.
#     7. **Input Data** – Example values used for the test.
#     8. **Expected Results** – What should happen after performing the test.

#     **Context:**
#     {context}

#     **Question:**
#     {question}

#     **Note:** Provide quick, actionable, and testable results focusing on the scheduling applications most important features. Test cases should focus on common tasks like task assignment, status updates, and resource handling.
#     """
# }

# # Load Ollama Embedding model for vector search
# embedder = OllamaEmbeddings(model='bge-m3')

# # Load vectors from the folder
# def load_vectors_from_folder(folder_path):
#     """Load all vector embeddings from the specified folder."""
#     vector_data = {}
#     for file_name in os.listdir(folder_path):
#         if file_name.endswith(".pkl"):
#             doc_name = file_name.split(".")[0]  # Get the name from file (without extension)
#             file_path = os.path.join(folder_path, file_name)
#             with open(file_path, "rb") as f:
#                 vector_data[doc_name] = pickle.load(f)
#     return vector_data

# # Search for relevant vectors based on the query
# def search_vectors(query, vector_data):
#     """Find the most relevant document vectors based on the query."""
#     query_embedding = embedder.embed_query(query)
    
#     best_match = None
#     best_score = float('-inf')

#     for doc_name, vectors in vector_data.items():
#         for vector in vectors:
#             similarity = sum(a * b for a, b in zip(query_embedding, vector))
#             if similarity > best_score:
#                 best_score = similarity
#                 best_match = doc_name
    
#     return best_match

# # Generate response using the tinyllama:latest model (or any specified version)
# def generate_ollama_response(prompt, model_name="tinyllama:latest"):
#     """Generate response using the Ollama model based on the prompt."""
#     # Ensure the Ollama API is used to generate the response with the specified model
#     response = ollama.chat(model=model_name, messages=[{"role": "user", "content": prompt}])
    
#     # Print the full response to inspect its structure
#     print("Full response:", response)

#     # Attempt to retrieve 'text' or check for other relevant keys in the response
#     if 'text' in response:
#         return response['text']
#     else:
#         # Check for alternative response structures
#         return response.get('message', 'No response text found')


# # Function to process user queries
# def process_query(query, detail_level="detailed"):
#     """Process user query and return response."""
#     query_lower = query.lower()
    
#     # Check for testing-related keywords
#     if any(keyword in query_lower for keyword in TESTING_KEYWORDS):
#         # Load vectors from the folder
#         vector_data = load_vectors_from_folder(vectorstore_folder)
        
#         # Search for relevant document
#         relevant_doc = search_vectors(query, vector_data)

#         prompt_template = PROMPTS.get(detail_level, PROMPTS["detailed"])

#         if relevant_doc:
#             # Generate a response using tinyllama:latest
#             response = generate_ollama_response(f"Relevant Document: {relevant_doc}\n{prompt_template.format(context=relevant_doc, question=query)}", model_name="tinyllama:latest")
#             return f"**Relevant Document:** {relevant_doc}\n\n**tinyllama:latest Response:**\n{response}"
#         else:
#             # No document found, generate based only on LLM knowledge
#             response = generate_ollama_response(f"Relevant Document: None\n{prompt_template.format(context='No relevant documents available. Generate based on general knowledge.', question=query)}", model_name="tinyllama:latest")
#             return f"**No relevant document found, using tinyllama:latest knowledge:**\n\n**tinyllama:latest Response:**\n{response}"
    
#     return "This query does not relate to testing or unit prtinyllama:latestocessing."

# # Main function to run the query
# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO)
    
#     user_query = input("Enter your query: ")
#     detail_level = input("Choose detail level (detailed/concise): ").strip().lower()
    
#     if detail_level not in {"detailed", "concise"}:
#         print("Invalid detail level. Using 'detailed' by default.")
#         detail_level = "detailed"
    
#     result = process_query(user_query, detail_level)
#     print(result)
import logging
import os
import pickle
import time
import logging
import streamlit as st
from langchain.embeddings import OllamaEmbeddings
from ollama import chat
from concurrent.futures import ThreadPoolExecutor

# Set up logging
logging.basicConfig(level=logging.INFO)

# Path to the folder containing the vector files
vectorstore_folder = 'C:/Users/Guna Shankar/Downloads/Temp/Vectorizers'

# Keywords to trigger vector search
TESTING_KEYWORDS = {"testing", "unit test", "unit processing", "test case"}

# Prompt templates (precomputed)
PROMPTS = {
    "detailed": """
    You are an expert in software testing for a scheduling application designed to handle various task management operations, such as task assignment, prioritization, and resource management. Your task is to generate comprehensive and well-documented test cases for the scheduling application's functionality, considering different test scenarios based on the provided feature.

    In your response, provide test cases that cover common functionalities like task assignment, task transitions (e.g., "On Hold"), resource allocation, and system visibility. Include all relevant sections such as test case IDs, scenarios, steps, expected results, and any pre-conditions.

    Each test case should cover the following aspects:
    1. **Test Case Scenario** – A short description of the feature being tested.
    2. **Test Case ID** – A unique identifier for each test case.
    3. **Test Case** – The name of the test case.
    4. **Test Case Description** – A detailed explanation of what is being tested.
    5. **Pre-condition** – Any requirements or setup needed before executing the test.
    6. **Test Steps** – The detailed steps to execute the test.
    7. **Input Data** – Example input data used for the test.
    8. **Expected Results** – The anticipated outcome of the test.

    **Context:**
    {context}

    **Question:**
    {question}
    """,
}

# Load Ollama Embedding model for vector search
embedder = OllamaEmbeddings(model='bge-m3')

# Load vectors from the folder
def load_vectors_from_folder(folder_path):
    vector_data = {}
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".pkl"):
            doc_name = file_name.split(".")[0]
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, "rb") as f:
                vector_data[doc_name] = pickle.load(f)
    return vector_data

# Search for relevant vectors based on the query (parallelized)
def search_vectors_parallel(query, vector_data):
    query_embedding = embedder.embed_query(query)

    def calculate_similarity(doc_name, vectors):
        best_score = float('-inf')
        best_match = None
        for vector in vectors:
            similarity = sum(a * b for a, b in zip(query_embedding, vector))
            if similarity > best_score:
                best_score = similarity
                best_match = doc_name
        return best_match

    with ThreadPoolExecutor() as executor:
        future_to_doc = {executor.submit(calculate_similarity, doc_name, vectors): doc_name for doc_name, vectors in vector_data.items()}
        best_match = None
        best_score = float('-inf')
        for future in future_to_doc:
            result = future.result()
            if result is not None:
                best_match = result
        return best_match

# Generate response using Llama3.2 from Ollama
def generate_ollama_response(prompt, model_name="llama3.2"):
    response = chat(model=model_name, messages=[{"role": "user", "content": prompt}])
    return response.message.content

# Cache responses to avoid redundant computations
def cache_response(query, response):
    with open("response_cache.pkl", "ab") as cache_file:
        pickle.dump((query, response), cache_file)

def get_cached_response(query):
    if os.path.exists("response_cache.pkl"):
        with open("response_cache.pkl", "rb") as cache_file:
            while True:
                try:
                    cached_query, cached_response = pickle.load(cache_file)
                    if cached_query == query:
                        return cached_response
                except EOFError:
                    break
    return None

# Function to process user queries and yield the complete response once done
def process_query(query, detail_level="detailed"):
    # Check for testing-related keywords
    query_lower = query.lower()
    if any(keyword in query_lower for keyword in TESTING_KEYWORDS):
        vector_data = load_vectors_from_folder(vectorstore_folder)
        
        # Check cache first
        cached_response = get_cached_response(query)
        if cached_response:
            return cached_response
        
        # Search for relevant document
        relevant_doc = search_vectors_parallel(query, vector_data)
        prompt_template = PROMPTS.get(detail_level, PROMPTS["detailed"])

        if relevant_doc:
            response = generate_ollama_response(f"Relevant Document: {relevant_doc}\n{prompt_template.format(context=relevant_doc, question=query)}", model_name="llama3.2")
        else:
            response = generate_ollama_response(f"Relevant Document: None\n{prompt_template.format(context='No relevant documents available. Generate based on general knowledge.', question=query)}", model_name="llama3.2")
        
        # Cache the response for future use
        cache_response(query, response)
        
        return response

    else:
        return "This query does not relate to testing or unit processing."

# Streamlit UI for displaying results
def main():
    st.title("Test Case Generator for Scheduling Application")

    # Input field for user query
    user_query = st.text_input("Enter your query:", "")

    # Input field for selecting detail level (concise/detailed)
    detail_level = st.radio("Choose detail level:", ["detailed", "concise"])

    # Button to generate test case
    if st.button("Generate Test Case"):
        if user_query:
            with st.spinner("Generating test case..."):
                result_placeholder = st.empty()

                # Get the final response
                result = process_query(user_query, detail_level)

                # Display the result
                result_placeholder.markdown(result)
        else:
            st.warning("Please enter a query to generate the test case.")

if __name__ == "__main__":
    main()
