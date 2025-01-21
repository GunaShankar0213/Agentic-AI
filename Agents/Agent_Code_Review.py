from phi.agent import Agent
from phi.model.groq import Groq

from dotenv import load_dotenv

# Placeholder for Code Agent (replace with actual code review logic)
code_review_agent = Agent(
    name="Code Agent",
    description="Reviews and analyzes code.",
    model=Groq(id="llama-3.3-70b-versatile"),
    instructions="Provide code suggestions and identify potential issues.",
    show_tool_calls=True,
    markdown=True,
    debug_mode=True
)