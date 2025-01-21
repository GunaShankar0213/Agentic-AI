from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.duckduckgo import DuckDuckGo
from dotenv import load_dotenv
from typing import Iterator
from phi.agent import RunResponse
from phi.utils.pprint import pprint_run_response

#pre process
import re

def preprocess_text(text):
    """
    Removes any type of markdown found in the given string.

    Args:
        text: The input string.

    Returns:
        The string with markdown removed.
    """

    # Remove markdown formatting
    text = re.sub(r'\*[^*]*\*', '', text)  # Remove bold
    text = re.sub(r'\_[^\_]*\_', '', text)  # Remove italics
    text = re.sub(r'`[^`]*`', '', text)  # Remove code blocks
    text = re.sub(r'\[[^]]*\]\([^)]*\)', '', text)  # Remove links
    text = re.sub(r'\#[a-zA-Z0-9-_]+', '', text)  # Remove hashtags
    text = re.sub(r'@\w+', '', text)  # Remove mentions
    text = re.sub(r'https?://\S+', '', text)  # Remove URLs
    text = re.sub(r'<[^>]*>', '', text)  # Remove HTML/XML tags
    # Remove continuous '#' characters
    text = re.sub(r'#+', '', text) 

    # Convert to single line and remove extra spaces
    text = " ".join(text.split())

    return text

# Routing Agent
query_classifier_agent = Agent(
    name="Query Classifier Agent",
    description="Classifies user queries for appropriate agent routing.",
    model=Groq(id="llama-3.3-70b-versatile"),
    instructions="Determine the most suitable agent for the given query.",
    show_tool_calls=True,
    markdown=True,
    debug_mode=True
)


def route_query(user_query):
    """
    Uses the query_classifier_agent to determine the most suitable agent for the given query
    and captures the agent's response in a variable.

    Args:
        user_query: The user's input query.

    Returns:
        The appropriate Agent object, or None if no suitable agent is found.
    """

    response_stream: Iterator[RunResponse] = query_classifier_agent.run(
        f"Based on the query '{user_query}', determine which agent should handle this: "
        "Web Agent, Code Review Agent, or Quantization Agent. "
        "Provide a concise answer without explanations.",
        stream=True,
        messages=True
    )
    extracted_text = ""
    # Extract text from the response messages
    for response in response_stream:  # Iterate through the response stream
        for message in response.messages:
            if isinstance(message.content, str):
                extracted_text += message.content + "\n"   # Add newline for readability

    # Types to Print the response and extracted text
    # pprint_run_response(response, markdown=True)
    # print(f"Extracted text: {extracted_text}")
    #print(response.content)

    val = preprocess_text(response.content)
    print(f"Extracted and processed text: {val}")

    return val

def run_query_selector(user_query):
    """
    Runs the query selector.

    Args:
        user_query: The user's input query (optional).
    """
    load_dotenv()
    
    if not user_query:
        user_query = input("Please enter your query: ")
        print(f"You entered: {user_query}")
    
    return route_query(user_query)


if __name__ == "__main__":
    # Example usage
    run_query_selector("what is the capital of Nepal?")