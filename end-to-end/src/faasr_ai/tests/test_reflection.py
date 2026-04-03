from __future__ import annotations

import json
import os
from dotenv import load_dotenv
from openai import OpenAI
from typing import Callable

from faasr_ai.agents.reflection import build_reflection_graph, make_initial_reflection_state

load_dotenv()
if "OPENAI_API_KEY" in os.environ and os.environ["OPENAI_API_KEY"]:
    os.environ["OPENAI_API_KEY"] = os.environ["OPENAI_API_KEY"].strip()

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
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


def reflection_node(llm: Callable[[str], str], workflow_tasks: list, user_request: str, max_reflections: int = 3):
    """Run the reflection agent with the given workflow tasks."""
    reflector = build_reflection_graph(llm)
    reflection_state = make_initial_reflection_state(
        workflow_tasks=workflow_tasks,
        user_request=user_request,
        workflows_folder="functions/",
        max_reflections=max_reflections
    )
    reflection_final = reflector.invoke(reflection_state)
    return reflection_final


def main():
    print("="*80)
    print("REFLECTION AGENT TEST")
    print("="*80)
    
    # Example workflow tasks (intentionally has some issues to demonstrate reflection)
    # You can replace this with actual output from workflow agent
    sample_workflow_tasks = [
                {
                    "task_id": "1",
                    "dependent_task_ids": [],
                    "instruction": "Download the NOAA GHCND dataset.",
                    "inputs": [],
                    "outputs": [
                    "ghcnd_dataset.json"
                    ]
                },
                {
                    "task_id": "2",
                    "dependent_task_ids": [
                    "1"
                    ],
                    "instruction": "Process the downloaded dataset to clean and handle missing values.",
                    "inputs": [
                    "ghcnd_dataset.json"
                    ],
                    "outputs": [
                    "cleaned_dataset.json"
                    ]
                },
                {
                    "task_id": "3",
                    "dependent_task_ids": [
                    "2"
                    ],
                    "instruction": "Extract relevant fields for precipitation, min and max temperature.",
                    "inputs": [
                    "cleaned_dataset.json"
                    ],
                    "outputs": [
                    "extracted_data.json"
                    ]
                },
                {
                    "task_id": "4",
                    "dependent_task_ids": [
                    "3"
                    ],
                    "instruction": "Generate visualizations for precipitation trends.",
                    "inputs": [
                    "extracted_data.json"
                    ],
                    "outputs": [
                    "precipitation_trends.png"
                    ]
                },
                {
                    "task_id": "5",
                    "dependent_task_ids": [
                    "3"
                    ],
                    "instruction": "Generate visualizations for minimum temperature trends.",
                    "inputs": [
                    "extracted_data.json"
                    ],
                    "outputs": [
                    "min_temperature_trends.png"
                    ]
                },
                {
                    "task_id": "6",
                    "dependent_task_ids": [
                    "3"
                    ],
                    "instruction": "Generate visualizations for maximum temperature trends.",
                    "inputs": [
                    "extracted_data.json"
                    ],
                    "outputs": [
                    "max_temperature_trends.png"
                    ]
                },
                {
                    "task_id": "7",
                    "dependent_task_ids": [
                    "4",
                    "5",
                    "6"
                    ],
                    "instruction": "Compile visualizations into a report or dashboard.",
                    "inputs": [
                    "precipitation_trends.png",
                    "min_temperature_trends.png",
                    "max_temperature_trends.png"
                    ],
                    "outputs": [
                    "final_report.pdf"
                    ]
                }
                ]
    
    user_request = """ The workflow should begin by downloading the NOAA Global Historical Climatology Network Daily (GHCND) dataset for station USC00351862. The process the 
    downloaded data, which is expected to be in a structured format such as CSV or JSON containing daily climate data. The transformation will involve 
    cleaning the data, handling any missing values, and extracting relevant fields for precipitation, minimum temperature, and maximum temperature. 
    The final output should be a series of visualizations that effectively represent the trends of precipitation, minimum temperature, and maximum temperature 
    over time, ideally in formats such as line graphs or bar charts, suitable for presentation in a report or dashboard. It is assumed that the dataset is 
    comprehensive and covers a significant time period to allow for meaningful analysis."""

    print(f"\n--- Initial Workflow Tasks ---")
    print(f"Total tasks: {len(sample_workflow_tasks)}")
    for task in sample_workflow_tasks:
        print(f"  [{task['task_id']}]: {task['instruction']}")
    
    print("\n" + "="*80)
    print("Running Reflection Agent...")
    print("="*80)
    
    try:
        # Run reflection agent
        max_reflections = 3
        
        final_state = reflection_node(
            llm_call,
            sample_workflow_tasks,
            user_request,
            max_reflections=max_reflections
        )
        
        # Display results
        print("\n" + "="*80)
        print("REFLECTION COMPLETE")
        print("="*80)
        
        # Show final validation result
        validation_result = final_state.get("validation_result", {})
        print(f"\nFinal Overall Quality: {validation_result.get('overall_quality', 'unknown')}")
        print(f"Reflection Iterations: {final_state.get('reflection_count', 0)}")
        print(json.dumps(final_state["workflow_tasks"], indent=2, ensure_ascii=False))
        
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()