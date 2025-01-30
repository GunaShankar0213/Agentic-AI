import streamlit as st
import subprocess
import logging
from main_gpt2 import process_query, load_gpt2_model_and_tokenizer

# Streamlit UI
def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Title for the app
    st.title("Test Case Generator for Scheduling Application")

    # Load the GPT-2 model and tokenizer (done only once)
    model, tokenizer = load_gpt2_model_and_tokenizer()

    if model and tokenizer:
        # Input field for user query
        user_query = st.text_input("Enter your query:", "")

        # Input field for selecting detail level (concise/detailed)
        detail_level = st.radio("Choose detail level:", ["detailed", "concise"])

        # Button to generate test case
        if st.button("Generate Test Case"):
            if user_query:
                with st.spinner("Generating test case..."):
                    # Process the query and generate the response
                    result = process_query(user_query, model, tokenizer, detail_level)
                    st.markdown(result)
            else:
                st.warning("Please enter a query to generate the test case.")
    else:
        st.error("Failed to load GPT-2 model. Please check your configuration.")

if __name__ == "__main__":
    main()
