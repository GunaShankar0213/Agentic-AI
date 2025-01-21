from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.duckduckgo import DuckDuckGo
from dotenv import load_dotenv
from Agents.Agent_Query_Selector import route_query  # Assuming route_query is defined elsewhere

# Import your custom agents (replace placeholders with actual implementations)
from Agents.Agent_web_search import web_search_agent
from Agents.Agent_Code_Review import code_review_agent
from Agents.Agent_Quantization import quantization_agent

def run_multiple_queries(queries):
    """
    Runs multiple queries through the appropriate agent based on query classification.

    Args:
        queries: A list of strings representing the user's queries.
    """

    for query in queries:
        # Use route_query to determine the most suitable agent
        selected_agent = route_query(query)

        if selected_agent is None:
            print(f"I'm not sure which agent to use for the query: '{query}'. Please try rephrasing. Still Using web to search")
            web_search_agent.print_response(query, stream=True)
            continue

        # Handle each agent type with appropriate print_response calls
        if selected_agent.name.lower().startswith("web"):
            web_search_agent.print_response(query, stream=True)
        elif selected_agent.name.lower().startswith("code"):
            code_review_agent.print_response(query, stream=True)
        elif selected_agent.name.lower().startswith("quantization"):
            quantization_agent.print_response(query, stream=True)
        else:
            print(f"An unexpected agent type was selected: {selected_agent.name}")

if __name__ == "__main__":
    queries = ["capital of India", "total sales of the product", "python code having fixes solve it def add ():"]
    run_multiple_queries(queries)