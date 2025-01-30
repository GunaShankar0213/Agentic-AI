import logging
import os
import pickle
import subprocess
from langchain.embeddings import OllamaEmbeddings
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Path to the folder containing the vector files
vectorstore_folder = 'C:/Users/Guna Shankar/Downloads/Temp/Vectorizers'

# Keywords to trigger vector search
TESTING_KEYWORDS = {"testing", "unit test", "unit processing", "test case"}

# Prompt templates
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

    Ensure that you offer examples of input data that will test the application in realistic and edge case scenarios. For example, test scenarios can involve normal task status changes or edge cases like assigning tasks to unavailable resources, handling conflicts between tasks, or scheduling tasks in overlapping time frames.

    **Context:**
    {context}

    **Question:**
    {question}

    **Guidance:**
    Provide highly detailed test cases to address different scenarios in the scheduling application, ensuring coverage of core functionalities. Each test case should be easy to follow, properly formatted, and aligned with the best practices in scheduling applications.
    """,
    
    "concise": """
    You are an expert in software testing for a scheduling application, and your role is to generate focused and clear test cases that cover core functionalities, such as task transitions, resource management, and task assignment. 

    Provide the test case structure in a brief format with:
    1. **Test Case Scenario** – A brief description of the test being tested.
    2. **Test Case ID** – A unique identifier.
    3. **Test Case** – The name of the test case.
    4. **Test Case Description** – A short explanation of the functionality being tested.
    5. **Pre-condition** – Any setup required before running the test.
    6. **Test Steps** – A concise list of actions to execute the test.
    7. **Input Data** – Example values used for the test.
    8. **Expected Results** – What should happen after performing the test.

    **Context:**
    {context}

    **Question:**
    {question}

    **Note:** Provide quick, actionable, and testable results focusing on the scheduling applications most important features. Test cases should focus on common tasks like task assignment, status updates, and resource handling.
    """
}

# Load GPT-2 model and tokenizer
def load_gpt2_model_and_tokenizer():
    """Load GPT-2 model and tokenizer."""
    try:
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        logging.info("GPT-2 model and tokenizer loaded successfully.")
        return model, tokenizer
    except Exception as e:
        logging.error(f"Error loading GPT-2 model: {str(e)}")
        return None, None

# Load embeddings model for vector search
embedder = OllamaEmbeddings(model='bge-m3')

# Load vectors from the folder
def load_vectors_from_folder(folder_path):
    """Load all vector embeddings from the specified folder."""
    vector_data = {}
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".pkl"):
            doc_name = file_name.split(".")[0]  # Get the name from file (without extension)
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, "rb") as f:
                vector_data[doc_name] = pickle.load(f)
    return vector_data

# Search for relevant vectors based on the query
def search_vectors(query, vector_data):
    """Find the most relevant document vectors based on the query."""
    query_embedding = embedder.embed_query(query)
    
    best_match = None
    best_score = float('-inf')

    for doc_name, vectors in vector_data.items():
        for vector in vectors:
            similarity = sum(a * b for a, b in zip(query_embedding, vector))
            if similarity > best_score:
                best_score = similarity
                best_match = doc_name
    
    return best_match

def generate_gpt2_response(prompt, model, tokenizer):
    """Generate response from GPT-2 model based on the prompt."""
    # Set the pad_token to eos_token
    tokenizer.pad_token = tokenizer.eos_token
    
    inputs = tokenizer.encode(prompt, return_tensors="pt", padding=True, truncation=True)
    
    # Increase max_new_tokens to handle longer input sequences
    outputs = model.generate(inputs, max_new_tokens=500, no_repeat_ngram_size=2)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response



# Function to process user queries
def process_query(query, model, tokenizer, detail_level="detailed"):
    """Process user query and return response."""
    query_lower = query.lower()
    
    # Check for testing-related keywords
    if any(keyword in query_lower for keyword in TESTING_KEYWORDS):
        # Load vectors from the folder
        vector_data = load_vectors_from_folder(vectorstore_folder)
        
        # Search for relevant document
        relevant_doc = search_vectors(query, vector_data)

        prompt_template = PROMPTS.get(detail_level, PROMPTS["detailed"])

        if relevant_doc:
            # Generate a response using GPT-2
            response = generate_gpt2_response(f"Relevant Document: {relevant_doc}\n{prompt_template.format(context=relevant_doc, question=query)}", model, tokenizer)
            return f"**Relevant Document:** {relevant_doc}\n\n**GPT-2 Response:**\n{response}"
        else:
            # No document found, generate based only on LLM knowledge
            response = generate_gpt2_response(f"Relevant Document: None\n{prompt_template.format(context='No relevant documents available. Generate based on general knowledge.', question=query)}", model, tokenizer)
            return f"**No relevant document found, using GPT-2 knowledge:**\n\n**GPT-2 Response:**\n{response}"
    
    return "This query does not relate to testing or unit processing."

# Main function to run the query
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Load GPT-2 model and tokenizer
    model, tokenizer = load_gpt2_model_and_tokenizer()
    
    if model and tokenizer:
        user_query = input("Enter your query: ")
        detail_level = input("Choose detail level (detailed/concise): ").strip().lower()
        
        if detail_level not in {"detailed", "concise"}:
            print("Invalid detail level. Using 'detailed' by default.")
            detail_level = "detailed"
        
        result = process_query(user_query, model, tokenizer, detail_level)
        print(result)
    else:
        print("Failed to load the GPT-2 model.")

## Inout: Can you generate test cases for scheduling tasks in a system that manages resources and assignments?