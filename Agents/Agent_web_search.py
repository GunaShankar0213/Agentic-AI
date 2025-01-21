from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.duckduckgo import DuckDuckGo
from dotenv import load_dotenv

# Define Agents
web_search_agent = Agent(
    name="Web Agent",
    description="Searches the web for general information.",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[DuckDuckGo()],
    instructions="Always include the source.",
    show_tool_calls=True,
    markdown=True,
    debug_mode=True
)