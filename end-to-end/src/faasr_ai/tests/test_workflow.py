from __future__ import annotations

import json
import os
from dotenv import load_dotenv
from openai import OpenAI
from faasr_ai.agents.workflow import workflow_agent_node, make_initial_workflow_state
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


def workflow_node(llm: Callable[[str], str], user_request: str):
    """Run the workflow agent with the given LLM and user request."""
    init_wf_state = make_initial_workflow_state(user_request)
    wf_final = workflow_agent_node(init_wf_state, llm_call)
    return wf_final


def main():
    # Get user question
    user_request = """
    The workflow should begin by downloading the NOAA Global Historical Climatology Network Daily (GHCND) dataset. The process the 
    downloaded data, which is expected to be in a structured format such as CSV or JSON containing daily climate data. The transformation will involve 
    cleaning the data, handling any missing values, and extracting relevant fields for precipitation, minimum temperature, and maximum temperature. 
    The final output should be a series of visualizations that effectively represent the trends of precipitation, minimum temperature, and maximum temperature 
    over time, ideally in formats such as line graphs or bar charts, suitable for presentation in a report or dashboard. It is assumed that the dataset is 
    comprehensive and covers a significant time period to allow for meaningful analysis.
    """
    
    print("\n" + "="*80)
    print("Running Workflow Agent...")
    print("="*80)
    
    # Run clarification
    final_state = workflow_node(llm_call, user_request)
    
    # Display results
    print("\n" + "="*80)
    print("WORKFLOW GENERATION COMPLETE")
    print("="*80)
    
    print(json.dumps(final_state.get("workflow_tasks", {}), indent=2, ensure_ascii=False))
    
    
    
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()