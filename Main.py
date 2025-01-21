from dotenv import load_dotenv
from Agents.Agent_Query_Selector import route_query

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
        print(f"Selected Agent: {selected_agent}")

        if selected_agent is None:
            print(f"I'm not sure which agent to use for the query: '{query}'. Please try rephrasing. Still Using web to search")
            web_search_agent.print_response(query, stream=True)
            continue

        # Handle each agent type with appropriate print_response calls
        if selected_agent.lower().startswith("web"):
            web_search_agent.print_response(query, stream=True)

        elif selected_agent.lower().startswith("code"):
            code_review_agent.print_response(query, stream=True)

        elif selected_agent.lower().startswith("quantization"):
            quantization_agent.print_response(query, stream=True)

        else:
            print(f"An unexpected agent type was selected: {selected_agent}")
            web_search_agent.print_response(query, stream=True)

if __name__ == "__main__":
    load_dotenv()
    # For multiple query process load the queries in the list
    queries = ["capital of India", "total sales of the product", "python code having fixes solve it def add ():"]
    # Example usage
    # queries = ["What is the capital of Nepal?"]
    run_multiple_queries(queries)