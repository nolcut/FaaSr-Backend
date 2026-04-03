from __future__ import annotations

import json
import os
from dotenv import load_dotenv
from openai import OpenAI
from faasr_ai.agents.clarification import build_clarification_graph, make_initial_clarification_state
from typing import Callable

load_dotenv()
if "OPENAI_API_KEY" in os.environ and os.environ["OPENAI_API_KEY"]:
    os.environ["OPENAI_API_KEY"] = os.environ["OPENAI_API_KEY"].strip()

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0"))

client = OpenAI()

def llm_call(prompt: str) -> str:
    """LLM adapter used by all nodes."""
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=OPENAI_TEMPERATURE,
        messages=[
            {"role": "system", "content": "Follow instructions precisely. Output valid JSON when asked."},
            {"role": "user", "content": prompt},
        ],
    )
    return (resp.choices[0].message.content or "").strip()


def clarification_node(llm: Callable[[str], str], user_question: str):
    """Run the clarification agent with the given LLM and user question."""
    clarifier = build_clarification_graph(llm)
    clar_state = make_initial_clarification_state(user_question)
    clar_final = clarifier.invoke(clar_state)
    return clar_final


def main():
    # Get user question
    user_question = "Pulls data from the NOAA Global Historical Climatology Network Daily (GHCND) dataset, process the data, and create a visualization for precipitation, min temperature, max temperature."
    if not user_question:
        user_question = "Generate two sample data sets and sum them up"
        print(f"Using default: '{user_question}'")
        
    print("User Request: ", user_question)
    
    print("\n" + "="*80)
    print("Running Clarification Agent...")
    print("="*80)
    
    # Run clarification
    final_state = clarification_node(llm_call, user_question)
    
    # Display results
    print("\n" + "="*80)
    print("CLARIFICATION COMPLETE")
    print("="*80)
    
    print("\n--- Intent ---")
    print(final_state.get("intent"))
    
    print("\n--- Structured Request ---")
    print(json.dumps(final_state.get("final_structured_request", {}), indent=2))
    
    print("\n--- Enhanced Natural Language Request ---")
    print(final_state.get("enhanced_natural_language_request"))
    
    print("\n--- User Approved ---")
    print(final_state.get("user_approved"))
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()