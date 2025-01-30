import streamlit as st
import logging
from main import process_query  # Import the process_query function from model.py
import time
import pickle
import os

# Streamlit UI
def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Title for the app
    st.title("Test Case Generator for Scheduling Application")

    # Input field for user query
    user_query = st.text_input("Enter your query:", "")

    # Input field for selecting detail level (concise/detailed)
    detail_level = st.radio("Choose detail level:", ["detailed", "concise"])

    # Button to generate test case
    if st.button("Generate Test Case"):
        if user_query:
            with st.spinner("Generating test case..."):
                # Placeholder for intermediate results
                result_placeholder = st.empty()
                
                # Get the final response from process_query
                result = process_query(user_query, detail_level)

                # If result is available, display it progressively
                if result:
                    # Check if the result is cached and use it
                    cached_result = get_cached_response(user_query)
                    if cached_result:
                        result = cached_result
                    else:
                        cache_response(user_query, result)
                    
                    # Display the result progressively
                    generated_text = ""
                    for i in range(0, len(result), 100):  # Display results in chunks
                        chunk = result[i:i+100]
                        generated_text += chunk  # Concatenate chunks progressively
                        result_placeholder.markdown(generated_text)  # Show intermediate results
                        time.sleep(0.5)  # Simulate a slight delay

                else:
                    st.warning("No response generated for the query.")
        else:
            st.warning("Please enter a query to generate the test case.")

def get_cached_response(query):
    """Fetch cached response if exists."""
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

def cache_response(query, response):
    """Cache the response for future use."""
    with open("response_cache.pkl", "ab") as cache_file:
        pickle.dump((query, response), cache_file)

if __name__ == "__main__":
    main()
