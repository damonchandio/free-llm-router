"""python examples/quickstart.py  (needs GROQ_API_KEY and/or GEMINI_API_KEY, or a local Ollama)"""
from free_llm_router import AIError, Router

ai = Router()
print(ai.status())
try:
    print(ai.ask("You are concise.", "Explain retrieval-augmented generation in one sentence."))
    print(ai.ask_json("Extract fields as JSON.", "Name: Ada Lovelace, role: mathematician"))
except AIError as e:
    print("No backend answered:", e)
