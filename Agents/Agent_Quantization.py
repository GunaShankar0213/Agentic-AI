from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.duckduckgo import DuckDuckGo
from dotenv import load_dotenv
from typing import Iterator

# Placeholder for Quantization Agent (replace with actual database and KB logic)
quantization_agent = Agent(
    name="Quantization Agent",
    description="Searches knowledge base, SQL database, and analyzes summaries.",
    model=Groq(id="llama-3.3-70b-versatile"),
    instructions="Determine the most suitable charts and provide the most relevant information with stats.",
    show_tool_calls=True,
    markdown=True,
    debug_mode=True
)